import rospy
from actionlib import SimpleActionServer

from tf2_geometry_msgs import PointStamped
import tf2_msgs.msg
from geometry_msgs.msg import TransformStamped, PolygonStamped, Polygon, Point
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_multiply, rotation_matrix
import tf2_ros
from std_msgs.msg import Int32

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
        self.point_viz_topic = "/point_viz"
        self.point_viz_pub = rospy.Publisher(
            self.point_viz_topic, PointStamped, queue_size=10
        )

        # transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.ref_link = "camera_link"
        self.depth_optical_link_frame = "optical_depth_link_frame"
        self.tag_ref = "camera_color_optical_frame"
        self.tag_frame = "tag_0"
        self.tag_adj_frame = "tag_adj"

        # landmarks
        self.lm_body = [
            Landmarks.RIGHT_SHOULDER,
            Landmarks.LEFT_SHOULDER,
            Landmarks.RIGHT_HIP,
            Landmarks.LEFT_HIP,
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

        # Rotation matrices
        self.x_unit = np.array([1, 0, 0])
        self.y_unit = np.array([0, 1, 0])
        self.z_unit = np.array([0, 0, 1])

        # transform pub
        self.tf_pub = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=3)
        self.torso_only_tf_pub = rospy.Publisher(
            "torso_tf", TransformStamped, queue_size=1
        )
        self.tag_adj_stamped_tf_pub = rospy.Publisher(
            "/tag_adj_tf", TransformStamped, queue_size=1
        )
 

        # polygon pub
        self.torso_polygon_pub = rospy.Publisher(
            "/torso_polygon", PolygonStamped, queue_size=1
        )

        # new iteration pub
        self.pose_updated_pub = rospy.Publisher("/pose_updated", Int32, queue_size=1)
        self.updates_counter = 0

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
                    
                    torso = self.pose_estimate_body(landmarks_list)
                    #rospy.loginfo(f"torso {torso}")

                    t = TransformStamped()
                    t.header.frame_id = "/camera_link"
                    t.header.stamp = rospy.Time.now()
                    t.child_frame_id = "torso"
                    t.transform.translation.x = torso[0]
                    t.transform.translation.y = torso[1]
                    t.transform.translation.z = torso[2]
                    t.transform.rotation.x = torso[3]
                    t.transform.rotation.y = torso[4]
                    t.transform.rotation.z = torso[5]
                    t.transform.rotation.w = torso[6]
                    tfm = tf2_msgs.msg.TFMessage([t])
                    self.tf_pub.publish(tfm)
                    self.torso_only_tf_pub.publish(t)

                    try: 
                        tag_trans = self.tf_buffer.lookup_transform(self.ref_link,self.tag_frame, rospy.Time())

                        tag_trans.header.stamp = rospy.Time.now()
                        tag_trans.header.frame_id = "tag_0"
                        tag_trans.child_frame_id = "tag_adj"

                        angz = -np.pi/2 
                        angy = -np.pi/2 
                        rot_z = rotation_matrix(angz, (0,0,1))
                        rot_y = rotation_matrix(angy, (0,1,0))

                        q_rotz = quaternion_from_matrix(rot_z)
                        q_roty = quaternion_from_matrix(rot_y)

                        q_new = quaternion_multiply(q_rotz, q_roty)

                        tag_trans.transform.translation.x = 0 
                        tag_trans.transform.translation.y = 0 
                        tag_trans.transform.translation.z = 0 

                        tag_trans.transform.rotation.x = q_new[0]
                        tag_trans.transform.rotation.y = q_new[1] 
                        tag_trans.transform.rotation.z = q_new[2] 
                        tag_trans.transform.rotation.w = q_new[3] 

                        #rospy.loginfo(f"tag trans {tag_trans}")
                        tfm = tf2_msgs.msg.TFMessage([tag_trans])
                        self.tf_pub.publish(tfm)
                        #self.tag_adj_stamped_tf_pub.publish(tag_trans)
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rospy.loginfo("could not find tf?")


                    # update topic to signal that count has incremented
                    self.updates_counter = self.updates_counter + 1
                    counter_msg = Int32()
                    counter_msg.data = self.updates_counter
                    self.pose_updated_pub.publish(self.updates_counter)

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

        # get landmarks coords
        for lm in self.lm_body:
            # transform landmark to 3D pose and transform coords to base link
            landmarks[lm.value]["point"] = self.transform_point_cameralink(
                self.landmark_to_3d(landmarks[lm.value]["point"])
            )
            self.point_viz_pub.publish(landmarks[lm.value]["point"])
            
        torso_pt_stamped, torso_polygon = self.get_torso_pts(landmarks)
        # returns a dictionary of the landmarks, their info, and their 3D coordinate
        unit_nrml_x = self.compute_x_normal(landmarks, torso_pt_stamped.point)
        # use shoulder vector as y unit normal
        unit_nrml_y = self.compute_y_normal(landmarks) 
        unit_nrml_z = np.cross(unit_nrml_x, unit_nrml_y)

        R = np.stack((unit_nrml_x, unit_nrml_y, unit_nrml_z)).T
        T = np.eye(4); T[:3, :3] = R
        q = quaternion_from_matrix(T)

        # rospy.loginfo(f"x {unit_nrml_x} y {unit_nrml_y} z {unit_nrml_z}")
        # rospy.loginfo(f"xy_dot {np.dot(unit_nrml_x,unit_nrml_y)}")
        # rospy.loginfo(f"xz_dot {np.dot(unit_nrml_x,unit_nrml_z)}")
        # rospy.loginfo(f"yz_dot {np.dot(unit_nrml_y,unit_nrml_z)}")

        #rospy.loginfo(f"{q}")
        torso_pose = [
            torso_pt_stamped.point.x,
            torso_pt_stamped.point.y,
            torso_pt_stamped.point.z,
            q[0],
            q[1],
            q[2],
            q[3],
        ]
        torso_pt_stamped.point.x = torso_pt_stamped.point.x - 0.2
        self.point_viz_pub.publish(torso_pt_stamped)


        return torso_pose

    def compute_euler_ang(self, target_v, unit_v):
        c = np.dot(target_v, unit_v) / np.linalg.norm(target_v) / np.linalg.norm(unit_v)
        angle = np.arccos(c)
        return angle

    def compute_x_normal(self, landmarks, torso_point):
        # takes in landmarks and finds unit normal of plane
        # second point is the center point for the vectors
        # to be "crossed" upon
        p_rhip = landmarks[Landmarks.RIGHT_HIP.value]["point"].point
        p_lhip = landmarks[Landmarks.LEFT_HIP.value]["point"].point
        p_rsh = landmarks[Landmarks.RIGHT_SHOULDER.value]["point"].point
        p_lsh = landmarks[Landmarks.LEFT_SHOULDER.value]["point"].point

        vector_1 = np.array(
            [
                p_lsh.x - torso_point.x,
                p_lsh.y - torso_point.y,
                p_lsh.z - torso_point.z,
            ]
        )
        # using another landmark
        vector_2 = np.array(
            [
                p_rsh.x - torso_point.x,
                p_rsh.y - torso_point.y,
                p_rsh.z - torso_point.z,
            ]
        )

        # might need to check direction is no into bed
        plane_normal = np.cross(vector_1, vector_2)
        plane_unit_normal = plane_normal / np.linalg.norm(plane_normal)
        return plane_unit_normal

    def compute_y_normal(self, landmarks):
        p_rsh = landmarks[Landmarks.RIGHT_SHOULDER.value]["point"].point
        p_lsh = landmarks[Landmarks.LEFT_SHOULDER.value]["point"].point

        vector_1 = np.array(
            [
                p_lsh.x - p_rsh.x,
                p_lsh.y - p_rsh.y,
                p_lsh.z - p_rsh.z,
            ]
        )

        # might need to check direction is no into bed
        y_unit_normal = vector_1 / np.linalg.norm(vector_1)
        return y_unit_normal

    def get_torso_pts(self, landmarks):
        left_shoulder = landmarks[Landmarks.LEFT_SHOULDER.value]["point"].point
        right_shoulder = landmarks[Landmarks.RIGHT_SHOULDER.value]["point"].point

        left_hip = landmarks[Landmarks.LEFT_HIP.value]["point"].point
        right_hip = landmarks[Landmarks.RIGHT_HIP.value]["point"].point

        mid_shoulder = Point()
        mid_shoulder.x = 0.5 * (left_shoulder.x + right_shoulder.x)
        mid_shoulder.y = 0.5 * (left_shoulder.y + right_shoulder.y)
        mid_shoulder.z = 0.5 * (left_shoulder.z + right_shoulder.z)

        mid_hip = Point()
        mid_hip.x = 0.5 * (left_hip.x + right_hip.x)
        mid_hip.y = 0.5 * (left_hip.y + right_hip.y)
        mid_hip.z = 0.5 * (left_hip.z + right_hip.z)

        medial_lateral = np.array([[right_shoulder.x - left_shoulder.x], [right_shoulder.y - left_shoulder.y], [right_shoulder.z - left_shoulder.z]])
        antero_posterior = np.array([[mid_hip.x - mid_shoulder.x], [mid_hip.y - mid_shoulder.y], [mid_hip.z - mid_shoulder.z]])
        mid_shoulder_pvec = np.array([[mid_shoulder.x], [mid_shoulder.y], [mid_shoulder.z]])

        chest_pvec = mid_shoulder_pvec + 0.25 * (antero_posterior - (antero_posterior.T @ medial_lateral) / (medial_lateral.T @ medial_lateral) * medial_lateral)
        torso_point = Point(x=chest_pvec[0, 0], y=chest_pvec[1, 0], z=chest_pvec[2, 0])

        torso_polygon = PolygonStamped()
        torso_polygon.header.frame_id = "/camera_link"
        torso_polygon.header.stamp = rospy.Time.now()
        torso_polygon.polygon.points = [
            left_shoulder,
            mid_shoulder,
            right_shoulder,
            torso_point,
            right_hip,
            mid_hip,
            left_hip,
        ]
        self.torso_polygon_pub.publish(torso_polygon)

        torso_point_stamped = PointStamped()
        torso_point_stamped.header.frame_id = landmarks[Landmarks.LEFT_SHOULDER.value][
            "point"
        ].header.frame_id
        torso_point_stamped.point = torso_point

        return torso_point_stamped, torso_polygon

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
