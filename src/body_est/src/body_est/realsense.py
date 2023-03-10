import rospy
from actionlib import SimpleActionServer

from tf2_geometry_msgs import PointStamped, PoseStamped
from tf.transformations import quaternion_from_euler
import tf2_ros

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import time
import math
import copy
from enum import Enum


class Landmarks(Enum):
    # assign the integers to the useful ones
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24


class PoseEstimation:
    def __init__(self):
        self.logger_name = "pose_estimation"
        rospy.init_node("body_est", anonymous=True)
        rospy.loginfo(
            "Starting the Perception pose estimation node",
            logger_name=self.logger_name,
        )
        self.got_new_depth = False
        self.got_intrinsics = False

        # Rviz visualization and published topic
        self.pose_viz_topic = "/pose_viz"
        self.pose_viz_pub = rospy.Publisher(
            self.pose_viz_topic, PoseStamped, queue_size=10
        )

        # transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.ref_link = "camera_link"
        self.depth_optical_link_frame = "optical_depth_link_frame"

        # landmarks
        self.lm_body = [
            Landmarks.RIGHT_SHOULDER,
            Landmarks.LEFT_SHOULDER,
            #            Landmarks.RIGHT_HIP,
            #            Landmarks.LEFT_HIP,
        ]

        # TODO need to set a threshold to determine if joint is visible
        self.landmark_vis_thresh = 0.9

        # OpenCv
        self.cv_bridge = CvBridge()
        self.depth_img = None
        self.rgb_img = None
        self.camera_intrinsics = rs.intrinsics()
        self.MM_TO_M = 1 / 1000
        time.sleep(1.5)

        # camera data subs
        self.get_camera_info()
        self.depth_sub = rospy.Subscriber(
            "camera/aligned_depth_to_color/image_raw", Image, self.depth_image_cb
        )
        self.image_sub = rospy.Subscriber(
            "camera/color/image_raw", Image, self.rgb_image_cb
        )

    def depth_image_cb(self, data):
        # store the latest depth image for processing
        try:
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(data, "16UC1")
            self.got_new_depth = True
        except CvBridgeError as e:
            rospy.loginfo(e)

    def rgb_image_cb(self, data):
        # store the latest color image for processing
        try:
            self.rgb_img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            # only execute when have updated instrinsics, depth, and rgb
            if self.got_new_depth == True and self.got_intrinsics == True:
                # set updated depth to be false to wait until new depth and rgb
                self.got_new_depth = False
                with mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                ) as pose:
                    self.rgb_img.flags.writeable = False
                    image = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    landmarks_list = results.pose_landmarks.landmark
                    landmarks = self.pose_estimate_body(landmarks_list)
                    rospy.loginfo(f"Landmarks {landmarks}")

                    # process landmarks a bit
                    self.pose_viz_pub.publish(
                        landmarks[Landmarks.RIGHT_SHOULDER.value]["pose"]
                    )
                    self.pose_viz_pub.publish(
                        landmarks[Landmarks.LEFT_SHOULDER.value]["pose"]
                    )
                    self.pose_viz_pub.publish(
                        landmarks[Landmarks.RIGHT_HIP.value]["pose"]
                    )
                    self.pose_viz_pub.publish(
                        landmarks[Landmarks.LEFT_HIP.value]["pose"]
                    )


        except CvBridgeError as e:
            rospy.loginfo(e)

    def get_camera_info(self):
        info = rospy.wait_for_message(
            "camera/aligned_depth_to_color/camera_info", CameraInfo
        )

        # plumb model is also known as brown conrady
        self.depth_optical_link_frame = info.header.frame_id
        if info.distortion_model == "plumb_bob":
            self.camera_intrinsics.model = rs.distortion.brown_conrady
        else:
            self.camera_intrinsics.model = rs.distortion.none
        # intrinsics given as 1D array
        self.camera_intrinsics.fx = info.K[0]
        self.camera_intrinsics.fy = info.K[4]
        self.camera_intrinsics.ppx = info.K[2]
        self.camera_intrinsics.ppy = info.K[5]
        self.camera_intrinsics.width = info.width
        self.camera_intrinsics.height = info.height
        rospy.loginfo(f"Got camera intrinsics {self.camera_intrinsics}")
        self.got_intrinsics = True

    def pose_estimate_body(self, landmarks_list):
        # dict {'landmark_id': {'mp_landmark': landmark, 'point': PointStamped}}
        landmarks = {}
        # get important landmarks
        for lm in self.lm_body:
            # add empty pose stamped since landmark position is not true
            landmark_data = landmarks_list[lm.value]

            # if one of the points are not visible then fail
            if landmark_data.visibility < self.landmark_vis_thresh:
                return

            lm_pose = PoseStamped()
            lm_pose.header.frame_id = self.depth_optical_link_frame
            lm_pose.pose.position.x = landmark_data.x
            lm_pose.pose.position.y = landmark_data.y
            lm_pose.pose.position.z = landmark_data.z
            landmarks[lm.value] = {
                "mp_landmark": landmark_data,
                "pose": lm_pose,
            }
        # rospy.loginfo(f"{landmarks}")

        # get landmarks coords
        for lm in self.lm_body:
            # transform landmark to 3D pose and transform coords to base link
            landmarks[lm.value]["pose"] = self.transform_pose_cameralink(
                self.landmark_to_3d(landmarks[lm.value]["pose"])
            )
        # returns a dictionary of the landmarks, their info, and their 3D coordinate
        unit_nrml_x = compute_unit_normal(landmarks)
        unit_nrml_y = np.array(-unit_nrml_x[1], unit_nrml_x[0], unit_nrml_x[2])
        unit_nrml_z = np.array(-unit_nrml_x[2], unit_nrml_x[1], unit_nrml_x[0])

        position = est_torso_pt(landmarks)

        x_unit = np.array([1, 0, 0])
        y_unit = np.array([0, 1, 0])
        z_unit = np.array([0, 0, 1])

        def compute_euler_ang(target_v, unit_v)
            c = np.dot(target_v, unit_v)/np.linalg.norm(target_v)/np.linalg.norm(unit_v)
            angle = np.arccos(np.clip(c, -1, 1))
            return angle
    
        alpha = compute_euler_ang(unit_nrml_x, x_unit) 
        beta = compute_euler_ang(unit_nrml_y, y_unit)
        gamma = compute_euler_ang(unit_nrml_z, z_unit)

        q = quaternion_from_euler(alpha, beta, gamma)
        torso_pose = [position.x, position.y, position.z, q.x, q.y, q.z, q.w]

        return landmarks

    def compute_unit_normal(self, landmarks):
        # takes in landmarks and finds unit normal of plane
        # second point is the center point for the vectors
        # to be "crossed" upon
        p1 = landmarks[Landmarks.LEFT_SHOULDER.value]["pose"].pose.position
        p2 = landmarks[Landmarks.RIGHT_SHOULDER.value]["pose"].pose.position
        p3 = landmarks[Landmarks.RIGHT_HIP.value]["pose"].pose.position

        vector_1 = np.array(
            [
                p1.x - p2.x,
                p1.y - p2.y,
                p1.z - p2.z,
            ]
        )
        # using another landmark
        vector_2 = np.array(
            [
                p3.x - p2.x,
                p3.y - p2.y,
                p3.z - p2.z,
            ]
        )

        # might need to check direction is no into bed
        plane_normal = np.cross(vector_1, vector_2)
        plane_unit_normal = plane_normal / np.linalg.norm(plane_normal)
        return plane_unit_normal

    def est_torso_pt(self, landmarks):
        # take in 3-4 points, use vectors to find the axes
        # average xyz values to get a point
        p1 = landmarks[Landmarks.LEFT_SHOULDER.value]["pose"].pose.position
        p2 = landmarks[Landmarks.RIGHT_SHOULDER.value]["pose"].pose.position
        p3 = landmarks[Landmarks.RIGHT_HIP.value]["pose"].pose.position
        p4 = landmarks[Landmarks.LEFT_HIP.value]["pose"].pose.position

        x = (p1.x + p2.x + p3.x + p4.x) / 4
        y = (p1.y + p2.y + p3.y + p4.y) / 4
        z = (p1.z + p2.z + p3.z + p4.z) / 4

        return 1

    def landmark_to_3d(self, pose_stamped):
        # Compute the 3D coordinate of each pose. 3D values in mm
        x_pixel = int(pose_stamped.pose.position.x * self.camera_intrinsics.width)
        y_pixel = int(pose_stamped.pose.position.y * self.camera_intrinsics.height)

        depth = self.depth_img[y_pixel][x_pixel]

        # librealsense 2 function see opencv_pointcloud_viewer.py
        # inverse projection
        point_vals = rs.rs2_deproject_pixel_to_point(
            self.camera_intrinsics, [x_pixel, y_pixel], depth
        )

        # deproject returns an array of points
        new_pose_stamped = PoseStamped()
        new_pose_stamped.header.frame_id = pose_stamped.header.frame_id
        new_pose_stamped.pose.position.x = point_vals[0] * self.MM_TO_M
        new_pose_stamped.pose.position.y = point_vals[1] * self.MM_TO_M
        new_pose_stamped.pose.position.z = point_vals[2] * self.MM_TO_M

        return new_pose_stamped

    def transform_pose_cameralink(self, pose_stamped):
        # transforms from the optical lens to the camera link
        # flips point into more reasonable coordinate frame
        while not rospy.is_shutdown():
            try:
                # rospy.loginfo(f"point before {point_stamped}")
                transformed_pose = self.tf_buffer.transform(
                    pose_stamped,
                    self.ref_link,  # rospy.Time()
                )
                # rospy.loginfo(f"point after {transformed_point}")
                return transformed_pose
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue
            rospy.sleep(0.2)


def main():
    server = PoseEstimation()
    rospy.spin()


if __name__ == "__main__":
    main()
