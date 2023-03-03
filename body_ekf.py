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

    def augment(self, vec3d):
        return np.concatenate((vec3d, [[1]]))
    
    def position(self, state):
        return state[:3]
    
    def lin_velocity(self, state):
        return state[3:6]
    
    def orientation(self, state):
        return state[6:10]
    
    def ang_velocity(self, state):
        return state[10:]

    def state_to_tf(self, state):
        tf = tf_conversions.transformation.quaternion_matrix(self.orientation(state))
        tf[:3, 3] = self.position(state)
        return tf

    def forward_model(self, state, dt):
        new_pos = self.position(state) + self.lin_velocity(state) * dt
        
        ang_vel = self.ang_velocity(state)
        orientation_differential = 0.5 * np.array([
            [0, -ang_vel[0], -ang_vel[1], -ang_vel[2]],
            [ang_vel[0], 0, ang_vel[2], -ang_vel[1]],
            [ang_vel[1], -ang_vel[2], 0, ang_vel[0]],
            [ang_vel[2], ang_vel[1], -ang_vel[0], 0]
        ])
        new_orientation = self.orientation(state) + orientation_differential @ self.orientation(state) * dt
        new_orientation /= np.sqrt(np.sum(np.square(new_orientation)))

        return np.concatenate((new_pos, self.lin_velocity(state), new_orientation, self.ang_velocity(state)))

    def forward_jacob(self):
        raise NotImplementedError()

    def measurement_model(self, state):
        world_to_body_tf = self.state_to_tf(state)

        right_shoulder_world = world_to_body_tf @ self.augment(self.right_shoulder)
        left_shoulder_world = world_to_body_tf @ self.augment(self.left_shoulder)
        right_hip_world = world_to_body_tf @ self.augment(self.right_hip)
        right_hip_world = world_to_body_tf @ self.augment(self.left_hip)

        return np.concatenate((right_shoulder_world, left_shoulder_world, right_hip_world, left_hip_world))

    def measurement_jacob(self):
        raise NotImplementedError()


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

