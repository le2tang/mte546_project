import numpy as np

from body_est.fit_anatomical_frame import BodyPolygon


class ValidateBodyPoints:
    def __init__(self, p_thresh=0.01):
        self.dist_mean = np.array([0.2, 0.15, 0.2, 0.38, 0.35]) 
        self.dist_std = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.thresh = -2 * np.log(np.sqrt(2 * np.pi) * p_thresh)

    def is_valid(self, points):
        left_shoulder = self.pt2vec(points[BodyPolygon.LEFT_SHOULDER.value])
        mid_shoulder = self.pt2vec(points[BodyPolygon.MID_SHOULDER.value])
        right_shoulder = self.pt2vec(points[BodyPolygon.RIGHT_SHOULDER.value])
        torso = self.pt2vec(points[BodyPolygon.TORSO.value])
        right_hip = self.pt2vec(points[BodyPolygon.RIGHT_HIP.value])
        mid_hip = self.pt2vec(points[BodyPolygon.MID_HIP.value])
        left_hip = self.pt2vec(points[BodyPolygon.LEFT_HIP.value])

        upper_body_pts = np.stack((left_shoulder, mid_shoulder, right_shoulder, mid_hip))
        torso_diff = upper_body_pts - torso
        torso_dist = np.sqrt(np.sum(np.square(torso_diff), axis=1))

        shoulder_dist = np.array([np.sqrt(np.sum(np.square(left_shoulder - right_shoulder)))])

        #shoulder_dist = np.array([[np.sqrt(np.sum(np.square(left_shoulder - right_shoulder)))]])
        #dist = np.concatenate((torso_dist, shoulder_dist))
        dist = np.append(torso_dist, shoulder_dist)

        print(dist)

        log_p = np.square((dist - self.dist_mean) / self.dist_std)
        return log_p > self.thresh

    def pt2vec(self, pt, shape=(3,1)):
        return np.reshape(np.array([pt.x, pt.y, pt.z]), shape)
