import numpy as np

from enum import Enum
from geometry_msgs.msg import Transform

import tf_conversions

class BodyPolygon(Enum):
    LEFT_SHOULDER = 0
    MID_SHOULDER = 1
    RIGHT_SHOULDER = 2
    TORSO = 3
    RIGHT_HIP = 4
    MID_HIP = 5
    LEFT_HIP = 6
          

class FitAnatomicalFrame:
    def get_tf(self, left_shoulder_pt, right_shoulder_pt, torso_pt):
        # returns a dictionary of the landmarks, their info, and their 3D coordinate
        unit_nrml_x = self.compute_x_normal(left_shoulder_pt, right_shoulder_pt, torso_pt)
        # use shoulder vector as y unit normal
        unit_nrml_y = self.compute_y_normal(left_shoulder_pt, right_shoulder_pt) 
        unit_nrml_z = np.cross(unit_nrml_x, unit_nrml_y)

        R = np.stack((unit_nrml_x, unit_nrml_y, unit_nrml_z)).T
        T = np.eye(4); T[:3, :3] = R
        q = tf_conversions.transformations.quaternion_from_matrix(T)

        body_tf = Transform()
        body_tf.translation = torso_pt
        body_tf.rotation.x = q[0]
        body_tf.rotation.y = q[1]
        body_tf.rotation.z = q[2]
        body_tf.rotation.w = q[3]

        return body_tf

    def compute_x_normal(self, left_shoulder_pt, right_shoulder_pt, torso_pt):
        left_vec = np.array(
            [
                left_shoulder_pt.x - torso_pt.x,
                left_shoulder_pt.y - torso_pt.y,
                left_shoulder_pt.z - torso_pt.z,
            ]
        )
        right_vec = np.array(
            [
                right_shoulder_pt.x - torso_pt.x,
                right_shoulder_pt.y - torso_pt.y,
                right_shoulder_pt.z - torso_pt.z,
            ]
        )

        anterior = np.cross(left_vec, right_vec)
        anterior = anterior / np.linalg.norm(anterior)
        return anterior

    def compute_y_normal(self, left_shoulder_pt, right_shoulder_pt):
        medial = np.array(
            [
                left_shoulder_pt.x - right_shoulder_pt.x,
                left_shoulder_pt.y - right_shoulder_pt.y,
                left_shoulder_pt.z - right_shoulder_pt.z,
            ]
        )

        # might need to check direction is no into bed
        medial = medial / np.linalg.norm(medial)
        return medial
