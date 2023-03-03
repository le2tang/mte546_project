import numpy as np
import tf_conversions

class BodyModel:
    def __init__(self):
        # Body frame:
        # origin: sternum
        # x: backwards to forwards
        # y: right to left
        # z: down to up

        # Body frame measurements
        self.right_shoulder = np.array([[0, -0.2, 0.2]]).T
        self.left_shoulder = np.array([[0, 0.2, 0.2]]).T
        self.right_hip = np.array([[0, -0.18, -0.35]]).T
        self.left_hip = np.array([[0, 0.18, 0.35]]).T

        self.dt = 0.1

    def forward_model(self, state):
        new_pos = self.__state_position(state) + self.__state_lin_velocity(state) * self.dt
        
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
        ang_vel = self.__vec2quat(self.__state_ang_velocity(state))
        orientation_differential = 0.5 * self.__quat_rmul_mat(ang_vel)
        new_orientation = self.__state_orientation(state) + orientation_differential @ self.__state_orientation(state) * self.dt
        new_orientation /= np.sqrt(np.sum(np.square(new_orientation)))

        return np.concatenate((new_pos, self.__state_lin_velocity(state), new_orientation, self.__state_ang_velocity(state)))

    def forward_jacob(self, state):
        jacobian = np.zeros((13, 13))

        # Derivative of position is velocity
        jacobian[:3, 3:6] = self.dt * np.eye(3)

        # Derivative of linear velocity is 0

        # Derivative of orientation is angular velocity
        ang_vel = self.__vec2quat(self.__state_ang_velocity(state))
        jacobian[6:10, 6:10] = 0.5 * self.dt * self.__quat_rmul_mat(ang_vel)
        jacobian[6:10, 10:] = self.__quat_lmul_mat(self.__state_orientation(state))[1:, :]

        # Derivative of angular velocity is 0

    def measurement_model(self, state):
        # Only use the first three rows since the last row is for homogeneous coordinates
        world_to_body_tf = self.__state_tf(state)[:3, :]

        right_shoulder_world = world_to_body_tf @ self.__augment(self.right_shoulder)
        left_shoulder_world = world_to_body_tf @ self.__augment(self.left_shoulder)
        right_hip_world = world_to_body_tf @ self.__augment(self.right_hip)
        left_hip_world = world_to_body_tf @ self.__augment(self.left_hip)

        return np.concatenate((right_shoulder_world, left_shoulder_world, right_hip_world, left_hip_world))

    def measurement_jacob(self, state):
        jacobian = np.zeros((12, 13))

        # Derivative with respect to position
        jacobian[:3, :3] = np.eye(3)
        jacobian[3:6, :3] = np.eye(3)
        jacobian[6:9, :3] = np.eye(3)
        jacobian[9:, :3] = np.eye(3)


        # Derivative with respect to orientation
        orientation = self.__state_orientation(state)
        jacobian[:3, 6:10] = self.__orientation_jacobian(self.right_shoulder, orientation)
        jacobian[3:6, 6:10] = self.__orientation_jacobian(self.left_shoulder, orientation)
        jacobian[6:9, 6:10] = self.__orientation_jacobian(self.right_hip, orientation)
        jacobian[9:, 6:10] = self.__orientation_jacobian(self.left_hip, orientation)

        return jacobian
    
    def __orientation_jacobian(self, point, orientation):
        # d(qpq^-1) / dq

        point = self.__vec2quat(point)
        q1 = self.__quat_mul(orientation, point)
        q2 = self.__quat_mul(point, self.__quat_conj(orientation))

        # Q(qp) * I^* + Q_hat(pq^-1)
        return self.__quat_lmul_mat(q1) @ self.__quat_eye_conj() + self.__quat_rmul_mat(q2)

    def __state_position(self, state):
        return state[:3]
    
    def __state_lin_velocity(self, state):
        return state[3:6]
    
    def __state_orientation(self, state):
        return state[6:10]
    
    def ang_velocity(self, state):
        return state[10:]

    def __augment(self, vec3d):
        return np.concatenate((vec3d, [[1]]))
    
    def __vec2quat(self, vec3d):
        return np.concatenate(([[0]], vec3d))

    def __state_tf(self, state):
        tf = tf_conversions.transformation.quaternion_matrix(self.__state_orientation(state))
        tf[:3, 3] = self.__state_position(state)
        return tf
    
    def __quat_lmul_mat(self, q):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        # q * p = __quat_lmul(q) * p = Q(q) * p
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]]
        ])

    def __quat_rmul_mat(self, p):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        # q * p = __quat_rmul_mat(p) * q = Q_hat(p) * q
        return np.array([
            [p[0], -p[1], -p[2], -p[3]],
            [p[1], p[0], p[3], -p[2]],
            [p[2], -p[3], p[0], p[1]],
            [p[3], p[2], -p[1], p[0]]
        ])
    
    def __quat_eye_conj(self):
        eye_conj = np.eye(4)
        eye_conj[1:, 1:] *= -1
        return eye_conj
    
    def __quat_mul(self, q, p):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        return self.__quat_lmul_mat(q) @ p

    def __quat_conj(self, q):
        qconj = np.copy(q)
        qconj[1:] *= -1

class EKF:
    def __init__(self, model):
        self.model = model
    
        self.state = np.zeros((13, 1)) # [x, v, q, w]
        self.state_covar = np.eye(13)

    def predict(self):
        self.state = self.model.forward_model(self.state)

        # Update state covariance
        raise NotImplementedError()

    def correct(self, measurement):
        self.model_meas = self.model.measurement_model(self.state)

        # Update state covariance from kalman gain
        raise NotImplementedError()

