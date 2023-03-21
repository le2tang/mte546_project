import numpy as np


class PoseModel:
    def __init__(self):
        self.dt = 0.1

    def forward_model(self, state):
        """Returns 13x1 state vector after 1 timestep [p v q w]"""
        new_pos = self._state_position(state) + self.dt * self._state_lin_velocity(
            state
        )
        q = self._state_orientation(state)
        w = self._state_ang_velocity(state)
        new_lin_vel = self._state_lin_velocity(state)
        new_quat = self._state_orientation(
            state
        ) + self.dt * 0.5 * self._quat_derivative(q, w)
        new_ang_vel = self._state_ang_velocity(state)

        return np.concatenate((new_pos, new_lin_vel, new_quat, new_ang_vel))

    def forward_jacobian(self, state):
        """Returns 13x13 jacobian matrix of the forward model d[p v q w]/d[p v q w]"""
        q = self._state_orientation(state)
        w = self._state_ang_velocity(state)
        jacobian = np.zeros((13, 13))
        jacobian[:3, :3] = np.eye(3)  # dp/dp
        jacobian[3:6, 3:6] = np.eye(3)  # dv/dv
        jacobian[:3, 3:6] = np.eye(3) * 0.5  # dp/dv
        jacobian[6:10, 6:10] = self._dq_dq(w)  # dq/dq
        jacobian[6:10, 10:13] = self._dq_dw(q)  # dq/dw
        jacobian[10:13, 10:13] = np.eye(3)  # dw/dw

        return jacobian

    def measurement_model(self, state):
        """Returns 7x1 state measurement vector [p q]"""
        return np.concatenate(
            (self._state_position(state), self._state_orientation(state))
        )

    def measurement_jacobian(self, state):
        """Returns 7x13 jacobian matrix of the measurement model d[p q]/d[p v q w]"""
        H = np.zeros((7, 13))
        H[:3, :3] = np.eye(3)
        H[3:7, 6:10] = np.eye(4)

        return H

    def _state_position(self, state):
        """Extract position from state vector"""
        return np.reshape(state[:3], (3, 1))

    def _state_lin_velocity(self, state):
        """Extract linear velocity from state vector"""
        return np.reshape(state[3:6], (3, 1))

    def _state_orientation(self, state):
        """Extract orientation from state vector in q = [x y z w] format"""
        return np.reshape(state[6:10], (4, 1))

    def _state_ang_velocity(self, state):
        """Extract angular velocity from state vector"""
        return np.reshape(state[10:], (3, 1))

    def _vec2quat(self, vec3d):
        """Convert 3x1 vector to a quaternion in q = [x y z w] format"""
        return np.concatenate((vec3d, [[0]]))

    def _quat_derivative(self, q, w):
        """
        qx qy qz qw
        q0 q1 q2 q3

        wx wy wz
        w0 w1 w2
        """
        q = q.ravel()
        w = w.ravel()
        return np.array(
            [
                [0+ w[2] * q[2] -w[1] * q[2]+ w[0] * q[3]],
                [-w[2] * q[0]+ 0+ w[0] * q[2]+ w[1] * q[3]],
                [w[1] * q[0]-w[0] * q[1]+ 0+ w[2] * q[3]],
                [-w[0] * q[0]+ -w[1] * q[1] -w[2] * q[2]+ 0],
            ]
        )

    def _dq_dq(self, w):
        """
        wx wy wz
        w0 w1 w2
        """
        w = w.ravel()
        return (
            self.dt
            * 0.5
            * np.array(
                [
                    [2 / self.dt, w[2], -w[1], w[0]],
                    [-w[2], 2 / self.dt, w[0], w[1]],
                    [w[1], -w[0], 2 / self.dt, w[2]],
                    [-w[0], -w[1], -w[2], 2 / self.dt],
                ]
            )
        )

    def _dq_dw(self, q):
        """
        qx qy qz qw
        q0 q1 q2 q3
        """
        q = q.ravel()
        return (
            self.dt
            * 0.5
            * np.array(
                [
                    [q[3], -q[2], q[1]],
                    [q[2], q[3], -q[0]],
                    [-q[1], q[0], q[3]],
                    [-q[0], -q[1], -q[2]],
                ]
            )
        )
