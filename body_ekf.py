import numpy as np

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
        new_pos = self._state_position(state) + self._state_lin_velocity(state) * self.dt
        
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
        ang_vel = self._vec2quat(self._state_ang_velocity(state))
        new_orientation = self._state_orientation(state) + self.dt * 0.5 * self._quat_rmul_mat(ang_vel) @ self._state_orientation(state)
        new_orientation /= np.sqrt(np.sum(np.square(new_orientation)))

        return np.concatenate((new_pos, self._state_lin_velocity(state), new_orientation, self._state_ang_velocity(state)))

    def forward_jacobian(self, state):
        jacobian = np.zeros((13, 13))

        # Derivative of position is velocity
        jacobian[:3, 3:6] = self.dt * np.eye(3)

        # Derivative of linear velocity is 0

        # Derivative of orientation is angular velocity
        ang_vel = self._vec2quat(self._state_ang_velocity(state))
        jacobian[6:10, 6:10] = self.dt * 0.5 * self._quat_rmul_mat(ang_vel)
        jacobian[6:10, 10:] = self._quat_lmul_mat(self._state_orientation(state))[:, 1:]

        # Derivative of angular velocity is 0

        return jacobian

    def measurement_model(self, state):
        # Only use the first three rows since the last row is for homogeneous coordinates
        world_to_body_tf = self._state_tf(state)[:3, :]

        right_shoulder_world = world_to_body_tf @ self._augment(self.right_shoulder)
        left_shoulder_world = world_to_body_tf @ self._augment(self.left_shoulder)
        right_hip_world = world_to_body_tf @ self._augment(self.right_hip)
        left_hip_world = world_to_body_tf @ self._augment(self.left_hip)

        return np.concatenate((right_shoulder_world, left_shoulder_world, right_hip_world, left_hip_world))

    def measurement_jacobian(self, state):
        jacobian = np.zeros((12, 13))

        # Derivative with respect to position
        jacobian[:3, :3] = np.eye(3)
        jacobian[3:6, :3] = np.eye(3)
        jacobian[6:9, :3] = np.eye(3)
        jacobian[9:, :3] = np.eye(3)

        # Derivative with respect to orientation
        orientation = self._state_orientation(state)
        jacobian[:3, 6:10] = self._orientation_jacobian(self.right_shoulder, orientation)[1:, :]
        jacobian[3:6, 6:10] = self._orientation_jacobian(self.left_shoulder, orientation)[1:, :]
        jacobian[6:9, 6:10] = self._orientation_jacobian(self.right_hip, orientation)[1:, :]
        jacobian[9:, 6:10] = self._orientation_jacobian(self.left_hip, orientation)[1:, :]

        return jacobian
    
    def _orientation_jacobian(self, point, orientation):
        # d(qpq^-1) / dq

        point = self._vec2quat(point)
        q1 = self._quat_mul(orientation, point)
        q2 = self._quat_mul(point, self._quat_conj(orientation))

        # Q(qp) * I^* + Q_hat(pq^-1)
        return self._quat_lmul_mat(q1) @ self._quat_eye_conj() + self._quat_rmul_mat(q2)

    def _state_position(self, state):
        return np.reshape(state[:3], (3, 1))
    
    def _state_lin_velocity(self, state):
        return np.reshape(state[3:6], (3, 1))
    
    def _state_orientation(self, state):
        return np.reshape(state[6:10], (4, 1))
    
    def _state_ang_velocity(self, state):
        return np.reshape(state[10:], (3, 1))

    def _augment(self, vec3d):
        return np.concatenate((vec3d, [[1]]))
    
    def _vec2quat(self, vec3d):
        return np.concatenate(([[0]], vec3d))

    def _state_tf(self, state):
        tf = np.zeros((4, 4))
        tf[:3, :3] = self._quat_rot_mat(self._state_orientation(state))
        tf[:3, 3] = self._state_position(state)
        return tf
    
    def _quat_lmul_mat(self, q):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        # q * p = _quat_lmul(q) * p = Q(q) * p
        q = q.ravel()
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]]
        ])

    def _quat_rmul_mat(self, p):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        # q * p = _quat_rmul_mat(p) * q = Q_hat(p) * q
        p = p.ravel()
        return np.array([
            [p[0], -p[1], -p[2], -p[3]],
            [p[1], p[0], p[3], -p[2]],
            [p[2], -p[3], p[0], p[1]],
            [p[3], p[2], -p[1], p[0]]
        ])
    
    def _quat_eye_conj(self):
        eye_conj = np.eye(4)
        eye_conj[1:, 1:] *= -1
        return eye_conj
    
    def _quat_mul(self, q, p):
        # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion
        return self._quat_lmul_mat(q) @ p

    def _quat_conj(self, q):
        qconj = np.copy(q)
        qconj[1:] *= -1
        return qconj
    
    def _quat_rot_mat(self, q):
        q = q.ravel()
        return np.array([
            [1-2*(q[2]*q[2]+q[3]*q[3]), 2*(q[1]*q[2]-q[3]*q[0]), 2*(q[1]*q[3]+q[2]*q[0])],
            [2*(q[1]*q[2]+q[3]*q[0]), 1-2*(q[1]*q[1]+q[3]*q[3]), 2*(q[2]*q[3]-q[1]*q[0])],
            [2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1-2*(q[1]*q[1]+q[2]*q[2])]
        ])

class EKF:
    def _init_(self, model):
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

if __name__ == "__main__":
    body_model = BodyModel()

    print("Indices")
    test_state = np.reshape(np.arange(13), (13, 1))
    print(f"Position: {body_model._state_position(test_state)}")
    print(f"Lin. Velocity: {body_model._state_lin_velocity(test_state)}")
    print(f"Orientation: {body_model._state_orientation(test_state)}")
    print(f"Ang. Velocity: {body_model._state_ang_velocity(test_state)}")

    print("Forward Model Linear Velocity")
    test_state = np.array([[
        0, 0, 0,
        1, 1, 1,
        1, 0, 0, 0,
        0, 0, 0]]).T
    print(f"New state: {body_model.forward_model(test_state)}")

    print("Forward Model Angular Velocity")
    test_state = np.array([[
        0, 0, 0,
        0, 0, 0,
        1, 0, 0, 0,
        0, 0, 1]]).T
    print(f"Init. Orientation: {body_model._quat_rot_mat(body_model._state_orientation(test_state))}")
    new_state = body_model.forward_model(test_state)
    print(f"New Orientation: {body_model._quat_rot_mat(body_model._state_orientation(new_state))}")
    print(f"New state: {new_state}")

    print("Forward Jacobian")
    test_state = np.array([[
        0, 0, 0,
        1, 0, 0,
        1, 0, 0, 0,
        0, 0, 0]]).T
    print(body_model.forward_jacobian(test_state))

    print("Measurement Model")


    print("Measuremtent Jacobian")
    print(body_model.measurement_jacobian(test_state))
