#!/usr/bin/env python

import rospy
import numpy as np
import tf2_ros
import tf_conversions

from geometry_msgs.msg import PolygonStamped, Transform, TransformStamped
from body_est.ekf import EKF
from body_est.fit_anatomical_frame import BodyPolygon, FitAnatomicalFrame
from body_est.validate_body_points import ValidateBodyPoints
from body_est.model_equations import PoseModel
import matplotlib.pyplot as plt


class ViconInterface:
    TF_REF_NAME = "vicon_world"
    CAMERA_NAME = "camera"
    BODY_NAME = "body"

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
        return self.ekf.state[:3].flatten()

    def get_orientation(self):
        # Return quaternion as [x y z w]
        return (self.ekf.state[6:10] / np.sqrt(np.sum(np.square(self.ekf.state[6:10])))).flatten()

    def get_k(self):
        return self.ekf.K_k

    def get_k_norm(self):
        return np.linalg.norm(self.ekf.K_k)

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

    def predict(self):
        self.ekf.predict()

    def correct(self, measurement):
        self.ekf.correct(measurement)


class BodyPoseNode:
    def __init__(self):
        rospy.init_node("body_pose_node")
        
        self.body_pts_sub = rospy.Subscriber("torso_polygon", PolygonStamped, self.update)

        self.ekf = EKFInterface()

        self.fit_anatomical_frame = FitAnatomicalFrame()
        self.validate_body_points = ValidateBodyPoints()

        self.vicon = ViconInterface()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # TODO fix thresholds or give better initial estimate since error
        # is very large at beginning which prevents anything from running
        self.dist_error_threshold = 10 
        self.ang_error_threshold = 2 
        self.dist_errs = []
        self.ang_errs = []
        
        self.measured = []
        self.filtered = []
        self.k = []

    def update(self, msg):
        # Get the a priori estimate of the current state
        self.ekf.predict()
        predict_tf = self.ekf.get_transform()

        body_pts = msg.polygon.points
        body_tf = self.fit_anatomical_frame.get_tf(
            body_pts[BodyPolygon.LEFT_SHOULDER.value],
            body_pts[BodyPolygon.RIGHT_SHOULDER.value],
            body_pts[BodyPolygon.TORSO.value],
        )

        print(f"res = {self.validate_meas(body_pts, predict_tf, body_tf)}")

        if self.validate_meas(body_pts, predict_tf, body_tf):
            measurement = np.array(
                [
                    body_tf.translation.x,
                    body_tf.translation.y,
                    body_tf.translation.z,
                    body_tf.rotation.x,
                    body_tf.rotation.y,
                    body_tf.rotation.z,
                    body_tf.rotation.w,
                ]
            )
            self.ekf.correct(measurement)
            self.measured.append(measurement)

        # Report the posterior estimate transform
        estimate_tf = TransformStamped()
        estimate_tf.header.stamp = rospy.Time.now()
        estimate_tf.header.frame_id = "camera_link"
        estimate_tf.child_frame_id = "body_est/body"
        estimate_tf.transform = self.ekf.get_transform()
        filtered = np.array(
            [
                estimate_tf.transform.translation.x,
                estimate_tf.transform.translation.y,
                estimate_tf.transform.translation.z,
                estimate_tf.transform.rotation.x,
                estimate_tf.transform.rotation.y,
                estimate_tf.transform.rotation.z,
                estimate_tf.transform.rotation.w,
            ]
        )
        self.filtered.append(filtered)
 
        #rospy.loginfo(f"body filtered {estimate_tf.transform}")

        # Broadcast Transform from EKF prediction
        self.tf_broadcaster.sendTransform(estimate_tf)

        # ground_tf = self.vicon.get_cam_to_body()

        # error_tf, (rot_axis, rot_angle) = self.get_error(
        #    ground_tf, dist_error, estimate_tf.transform
        # )
        self.k.append(self.ekf.get_k_norm())

    def validate_meas(self, body_pts, predict_tf, body_tf):
        # Compare the measurement to the prior estimate
        error_tf, dist_error, (rot_angle, rot_axis) = self.get_error(
            predict_tf, body_tf
        )
        # Ignore the measurement if the difference is too large
        rospy.loginfo(f"dist err {dist_error} ang err {rot_angle}")
        small_prior_error = (dist_error < self.dist_error_threshold) and (
            rot_angle.all() < self.ang_error_threshold)

        body_points_valid = self.validate_body_points.is_valid(body_pts)

        print(f"small prior err {small_prior_error}")
        print(f"body pts valid{body_points_valid}")

        return small_prior_error and np.all(body_points_valid)

    def get_error(self, ref_tf, link_tf):
        error_tf = Transform()

        error_tf.translation.x = link_tf.translation.x - ref_tf.translation.x
        error_tf.translation.y = link_tf.translation.y - ref_tf.translation.y
        error_tf.translation.z = link_tf.translation.z - ref_tf.translation.z

        dist_error = np.sqrt(
            np.sum(
                np.square(
                    [
                        error_tf.translation.x,
                        error_tf.translation.y,
                        error_tf.translation.z,
                    ]
                )
            )
        )

        ref_rot_mat = tf_conversions.transformations.quaternion_matrix(
            [
                ref_tf.rotation.x,
                ref_tf.rotation.y,
                ref_tf.rotation.z,
                ref_tf.rotation.w,
            ]
        )
        link_rot_mat = tf_conversions.transformations.quaternion_matrix(
            [
                link_tf.rotation.x,
                link_tf.rotation.y,
                link_tf.rotation.z,
                link_tf.rotation.w,
            ]
        )

        error_rot_mat = link_rot_mat @ ref_rot_mat.T
        error_quat = tf_conversions.transformations.quaternion_from_matrix(
            error_rot_mat
        )
        error_quat /= np.sqrt(np.sum(np.square(error_quat)))

        error_tf.rotation.x = error_quat[0]
        error_tf.rotation.y = error_quat[1]
        error_tf.rotation.z = error_quat[2]
        error_tf.rotation.w = error_quat[3]

        rot_angle = 2 * np.arccos(error_quat[3])
        rot_axis = error_quat[:3] / np.sqrt(1 - error_quat[3] * error_quat[3])

        return error_tf, dist_error, (rot_angle, rot_axis)

    def write_to_file(self):
        plt.plot(self.k)
        plt.xlabel("Iteration #")
        plt.ylabel("K gain norm val")
        plt.title("K gain norm trend")
        plt.savefig("/home/felix/mte546/mte546_project/k_gain.png")

        self.measured = np.array(self.measured)
        self.filtered = np.array(self.filtered)
        rospy.loginfo("x y z qz qy qz qw")
        rospy.loginfo(f"measured var {np.var(self.measured, axis=0)}")
        rospy.loginfo(f"filtered var {np.var(self.filtered, axis=0)}")
        rospy.loginfo(f"measured max delta {np.max(self.measured, axis=0) - np.min(self.measured, axis=0)}")
        rospy.loginfo(f"filtered max delta {np.max(self.filtered, axis=0) - np.min(self.filtered, axis=0)}")


        self.measured_diff = np.diff(self.measured, axis=0)
        self.filtered_diff = np.diff(self.filtered, axis=0)

        # plot position trends
        plt.clf()
        plt.plot(self.measured[:, 0], label="Meas pX")
        plt.plot(self.measured[:, 1], label="Meas pY")
        plt.plot(self.measured[:, 2], label="Meas pZ")
        plt.plot(self.filtered[:, 0], label="filtered pX")
        plt.plot(self.filtered[:, 1], label="filtered pY")
        plt.plot(self.filtered[:, 2], label="filtered pZ")
        plt.legend()
        plt.xlabel("Iteration #")
        plt.ylabel("value (m)")
        plt.title("Positions")
        plt.savefig("/home/felix/mte546/mte546_project/positions.png")

        # plot quaterion trends
        plt.clf()
        plt.plot(self.measured[:, 3], label="Meas qX")
        plt.plot(self.measured[:, 4], label="Meas qY")
        plt.plot(self.measured[:, 5], label="Meas qZ")
        plt.plot(self.measured[:, 6], label="Meas qW")
        plt.plot(self.filtered[:, 3], label="filtered qX")
        plt.plot(self.filtered[:, 4], label="filtered qY")
        plt.plot(self.filtered[:, 5], label="filtered qZ")
        plt.plot(self.filtered[:, 6], label="filtered qW")
        plt.legend()
        plt.xlabel("Iteration #")
        plt.ylabel("value (rad)")
        plt.title("Quaterions")
        plt.savefig("/home/felix/mte546/mte546_project/quat.png")

        # plot position diff trends
        plt.clf()
        plt.plot(self.measured_diff[:, 0], label="Meas pX")
        plt.plot(self.measured_diff[:, 1], label="Meas pY")
        plt.plot(self.measured_diff[:, 2], label="Meas pZ")
        plt.plot(self.filtered_diff[:, 0], label="filtered pX")
        plt.plot(self.filtered_diff[:, 1], label="filtered pY")
        plt.plot(self.filtered_diff[:, 2], label="filtered pZ")
        plt.legend()
        plt.xlabel("Iteration #")
        plt.ylabel("value (m)")
        plt.title("Position deltas")
        plt.savefig("/home/felix/mte546/mte546_project/position_deltas.png")

        # plot quaterion trends
        plt.clf()
        plt.plot(self.measured_diff[:, 3], label="Meas qX")
        plt.plot(self.measured_diff[:, 4], label="Meas qY")
        plt.plot(self.measured_diff[:, 5], label="Meas qZ")
        plt.plot(self.measured_diff[:, 6], label="Meas qW")
        plt.plot(self.filtered_diff[:, 3], label="filtered qX")
        plt.plot(self.filtered_diff[:, 4], label="filtered qY")
        plt.plot(self.filtered_diff[:, 5], label="filtered qZ")
        plt.plot(self.filtered_diff[:, 6], label="filtered qW")
        plt.legend()
        plt.xlabel("Iteration #")
        plt.ylabel("value (rad)")
        plt.title("Quaterion deltas")
        plt.savefig("/home/felix/mte546/mte546_project/quat_deltas.png")


if __name__ == "__main__":
    body_pose_node = BodyPoseNode()
    rospy.spin()
    rospy.on_shutdown(body_pose_node.write_to_file)
