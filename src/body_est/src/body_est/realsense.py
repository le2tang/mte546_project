import rospy
from actionlib import SimpleActionServer

from tf2_geometry_msgs import PointStamped
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
        rospy.init_node("pose_estimation_server", anonymous=True)
        rospy.loginfo(
            "Starting the Perception pose estimation node",
            logger_name=self.logger_name,
        )

        # Rviz visualization
        self.point_viz_topic = "/point_viz"
        self.point_viz_pub = rospy.Publisher(
            self.point_viz_topic, PointStamped, queue_size=10
        )
        # TODO need a publisher for EKF to take in measurement

        # transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.ref_link = "camera_link"
        self.depth_optical_link_frame = "optical_depth_link_frame"

        # landmarks
        self.lm_body = [
            Landmarks.RIGHT_SHOULDER,
            Landmarks.LEFT_SHOULDER,
            Landmarks.RIGHT_EYE,
        ]
        self.lm_face = [
            Landmarks.LEFT_EYE,
            Landmarks.MOUTH_RIGHT,
            Landmarks.NOSE,
            Landmarks.MOUTH_LEFT,
            Landmarks.MOUTH_RIGHT,
        ]
        # TODO need to set a threshold to determine if joint is visible
        self.landmark_vis_thresh = 0.9

        # OpenCv
        self.cv_bridge = CvBridge()
        self.depth_img = None
        self.rgb_img = None
        self.camera_intrinsics = rs.intrinsics()
        self.MM_TO_M = 1 / 1000

        self.get_camera_info()

        # camera data subs
        self.depth_sub = rospy.Subscriber(
            "camera/aligned_depth_to_color/image_raw", Image, self.depth_image_cb
        )
        self.image_sub = rospy.Subscriber(
            "camera/color/image_raw", Image, self.rgb_image_cb
        )

        self.got_new_depth = False
        self.got_intrinsics = False

    def depth_image_cb(self, data):
        # store the latest depth image for processing
        try:
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(data, "16UC1")
            self.got_new_rgb = True
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
                    # publish to node?

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

            lm_point = PointStamped()
            lm_point.header.frame_id = self.depth_optical_link_frame
            lm_point.point.x = landmark_data.x
            lm_point.point.y = landmark_data.y
            lm_point.point.z = landmark_data.z
            landmarks[lm.value] = {
                "mp_landmark": landmark_data,
                "point": lm_point,
            }
        # rospy.loginfo(f"{landmarks}")

        # get landmarks coords
        for lm in self.lm_body:
            # transform landmark to 3D pose and transform coords to base link
            landmarks[lm.value]["point"] = self.transform_point_cameralink(
                self.landmark_to_3d(landmarks[lm.value]["point"])
            )
            # visualize points in Rviz
            self.point_viz_pub.publish(landmarks[lm.value]["point"])

        # rospy.loginfo(f"Transformed {landmarks}")

        # returns a dictionary of the landmarks, their info, and their 3D coordinate
        return landmarks

    def compute_unit_normal(self, landmarks, align_vertical=False):
        # takes in landmarks and finds unit normal of plane
        # second point is the center point for the vectors
        # to be "crossed" upon
        p1 = landmarks[Landmarks.LEFT_SHOULDER.value]["point"].point
        p2 = landmarks[Landmarks.RIGHT_SHOULDER.value]["point"].point
        p3 = landmarks[Landmarks.RIGHT_EYE.value]["point"].point

        vector_1 = np.array(
            [
                p1.x - p2.x,
                p1.y - p2.y,
                p1.z - p2.z,
            ]
        )

        # rospy.loginfo(f"Align vertical {align_vertical}")
        vector_2 = None
        if align_vertical == True:
            # place point directly vertical in Z direction to constrain the plane
            # to be vertical. using shoulders and a synthetic vertical point
            forced_v_point = copy.deepcopy(
                landmarks[Landmarks.RIGHT_SHOULDER.value]["point"]
            )
            forced_v_point.point.z += 0.2  # add arbitrary value to get a vertical plane
            rospy.loginfo(f"Forced point {forced_v_point}")
            self.point_viz_pub.publish(forced_v_point)

            vector_2 = np.array(
                [
                    forced_v_point.point.x - p2.x,
                    forced_v_point.point.y - p2.y,
                    forced_v_point.point.z - p2.z,
                ]
            )
        else:
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

    def landmark_to_3d(self, point_stamped):
        # Compute the 3D coordinate of each pose. 3D values in mm
        x_pixel = int(point_stamped.point.x * self.camera_intrinsics.width)
        y_pixel = int(point_stamped.point.y * self.camera_intrinsics.height)

        depth = self.depth_img[y_pixel][x_pixel]

        # librealsense 2 function see opencv_pointcloud_viewer.py
        # inverse projection
        point_vals = rs.rs2_deproject_pixel_to_point(
            self.camera_intrinsics, [x_pixel, y_pixel], depth
        )

        # deproject returns an array of points
        new_point_stamped = PointStamped()
        new_point_stamped.header.frame_id = point_stamped.header.frame_id
        new_point_stamped.point.x = point_vals[0] * self.MM_TO_M
        new_point_stamped.point.y = point_vals[1] * self.MM_TO_M
        new_point_stamped.point.z = point_vals[2] * self.MM_TO_M

        return new_point_stamped

    def transform_point_cameralink(self, point_stamped):
        # transforms from the optical lens to the camera link
        # flips point into more reasonable coordinate frame
        while not rospy.is_shutdown():
            try:
                # rospy.loginfo(f"point before {point_stamped}")
                transformed_point = self.tf_buffer.transform(
                    point_stamped,
                    self.ref_link,  # rospy.Time()
                )
                # rospy.loginfo(f"point after {transformed_point}")
                return transformed_point
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
