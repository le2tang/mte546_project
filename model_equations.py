import numpy as np


class PoseModel:
    def __init__(self):
        self.dt = 0.1

    def forward_model(self, state):
        """Returns 13x1 state vector after 1 timestep [p v q w]"""
        new_pos = self._state_position(state) + self.dt * self._state_lin_velocity(
            state
        )

        raise NotImplementedError()

    def forward_jacobian(self, state):
        """Returns 13x13 jacobian matrix of the forward model d[p v q w]/d[p v q w]"""
        raise NotImplementedError()

    def measurement_model(self, state):
        """Returns 7x1 state measurement vector [p q]"""
        return np.concatenate((self._state_position, self._state_orientation))

    def measurement_jacobian(self, state):
        """Returns 7x13 jacobian matrix of the measurement model d[p q]/d[p v q w]"""
        raise NotImplementedError()

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
