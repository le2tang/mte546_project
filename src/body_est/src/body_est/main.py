#!/usr/bin/env python

import rospy
import tf2
import tf_conversions

from actionlib import SimpleActionClient
from geometry_msgs.msg import Transform, StampedTransform
from body_ekf import EKF


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


class RealsenseInterface:
    def __init__(self):
        self.client = ServiceProxy()
        rospy.wait_for_service("")

    def get_body_pose(self):
        try:
            self.client()
        except:
            pass


class EKFInterface:
    def __init__(self):
        self.ekf = EKF()

    def get_position(self):
        return self.ekf.state[:3]

    def get_orientation(self):
        # Return quaternion as [x y z w]
        return self.ekf.state[6:10]

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

        self.ekf = EKFInterface()
        self.realsense = RealsenseInterface()

        self.vicon = ViconInterface()
        self.tf_broadcaster = tf2.TransformBroadcaster()

    def run(self):
        measurement = self.realsense.get_body_pose()

        self.ekf.update(measurement)

        estimate_tf = StampedTransform()
        estimate_tf.header.stamp = rospy.Time.now()
        estimate_tf.header.frame_id = "/bodypose/camera"
        estimate_tf.child_frame_id = "/bodypose/body"
        estimate_tf.transform = self.ekf.get_transform()

        # Broadcast Transform from EKF prediction
        self.tf_broadcaster.sendTransform(stamped_tf)

        self.compare(self, estimate_tf)

    def compare(self, estimate_tf):
        ground_tf = self.vicon.get_cam_to_body()

        position_error = np.array(
            [
                ground_tf.transform.position.x - estimate_tf.transform.position.x,
                ground_tf.transform.position.y - estimate_tf.transform.position.y,
                ground_tf.transform.position.z - estimate_tf.transform.position.z,
            ]
        )
        position_distance = np.sqrt(np.sum(np.square(position_error)))


if __name__ == "__main__":
    body_pose_node = BodyPoseNode()
    rospy.spin()
