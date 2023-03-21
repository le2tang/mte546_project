#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf_conversions

from geometry_msgs.msg import Transform, TransformStamped
from body_est.ekf import EKF
from body_est.model_equations import PoseModel


class ViconInterface:
    TF_REF_NAME = "/vicon_world"
    CAMERA_NAME = "/camera"
    BODY_NAME = "/body"

    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_cam_to_body(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.CAMERA_NAME, self.BODY_NAME, rospy.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(
                f"Transform lookup from {self.CAMERA_NAME} to {self.BODY_NAME} failed:\n{e}"
            )
            return None


class EKFInterface:
    def __init__(self):
        self.ekf = EKF(PoseModel())

    def get_position(self):
        return self.ekf.state[:3]

    def get_orientation(self):
        # Return quaternion as [x y z w]
        return self.ekf.state[6:10] / np.sqrt(np.sum(np.square(self.ekf.state[6:10])))

    def get_transform(self):
        position = self.get_position()
        orientation = self.get_orientation()

        tf = Transform()
        tf.translation.x = position[0]
        tf.translation.y = position[1]
        tf.translation.z = position[2]
        tf.rotation.x = orientation[0]
        tf.rotation.y = orientation[1]
        tf.rotation.z = orientation[2]
        tf.rotation.w = orientation[3]
        return tf

    def update(self, measurement):
        self.ekf.predict()
        self.ekf.correct(measurement)


class BodyPoseNode:
    def __init__(self):
        rospy.init_node("body_pose_node")

        self.realsense_sub = rospy.Subscriber("torso_tf", TransformStamped, self.update)

        self.ekf = EKFInterface()

        self.vicon = ViconInterface()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def get_error(self, ref_tf, link_tf):
        error_tf = Transform()

        error_tf.translation.x = link_tf.translation.x - ref_tf.translation.x
        error_tf.translation.y = link_tf.translation.y - ref_tf.translation.y
        error_tf.translation.z = link_tf.translation.z - ref_tf.translation.z

        ref_rot_mat = tf_conversions.transformations.quaternion_matrix(
            [
                ref_tf.orientation.x,
                ref_tf.orientation.y,
                ref_tf.orientation.z,
                ref_tf.orientation.z,
            ]
        )
        link_rot_mat = tf_conversions.transformations.quaternion_matrix(
            [
                link_tf.orientation.x,
                link_tf.orientation.y,
                link_tf.orientation.z,
                link_tf.orientation.z,
            ]
        )

        error_rot_mat = link_rot_mat @ ref_rot_mat.T
        error_quat = tf_conversions.transformations.quaternion_from_matrix(
            error_rot_mat
        )
        error_quat /= np.sqrt(np.sum(np.square(error_quat)))

        error_tf.orientation.x = error_quat[0]
        error_tf.orientation.y = error_quat[1]
        error_tf.orientation.z = error_quat[2]
        error_tf.orientation.w = error_quat[3]

        rot_angle = 2 * np.arccos(error_quat[3])
        rot_axis = error_quat[:3] / np.sqrt(1 - error_quat[3] * error_quat[3])

        return error_tf, (rot_angle, rot_axis)

    def update(self, msg):
        measurement = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w])

        self.ekf.update(measurement)

        estimate_tf = TransformStamped()
        estimate_tf.header.stamp = rospy.Time.now()
        estimate_tf.header.frame_id = "camera_link"
        estimate_tf.child_frame_id = "body_est/body"
        estimate_tf.transform = self.ekf.get_transform()

        # Broadcast Transform from EKF prediction
        self.tf_broadcaster.sendTransform(estimate_tf)

        #ground_tf = self.vicon.get_cam_to_body()

        #error_tf, (rot_axis, rot_angle) = self.get_error(
        #    ground_tf, estimate_tf.transform
        #)


if __name__ == "__main__":
    body_pose_node = BodyPoseNode()
    rospy.spin()
