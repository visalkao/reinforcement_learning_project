import gymnasium as gym
from typing import Optional, Union
from gymnasium import spaces
import numpy as np
import math
import random
from Utils import *
from collections import namedtuple
from Renderer import ArmRenderer
from scipy.interpolate import CubicSpline

SEED = 19930515
MAX_EPISODE_STEPS = 2000
FIXED_TARGET = True

Armconfig = namedtuple('Armconfig', ['SIZE_HUMERUS', 'WIDTH_HUMERUS', 'SIZE_RADIUS', 'WIDTH_RADIUS'])

class ArmReachingEnv2DTheta(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # Reset the environment with a fixed seed
        super().reset(seed=SEED)
        # Biomechanical parameters
        self.F_max = 10             # Maximum muscle force 
        self.r_shoulder = 0.05      # Shoulder moment arm
        self.r_elbow = 0.03         # Elbow moment arm 
        self.I_shoulder = 0.1       # Shoulder moment of inertia 
        self.I_elbow = 0.05         # Elbow moment of inertia 
        self.b = 0.1                # Viscous damping coefficient 
        self.dt = 0.05              # Time step 
        
        high_action_range = np.array(
            [1, 1, 1, 1, 1, 1], dtype=np.float32
        )
        self.action_space = spaces.Box(0, high_action_range, dtype=np.float32)

        # Modified observation space: 8 dimensions:
        # [theta_1, theta_2, omega_shoulder, omega_elbow, alpha_shoulder, alpha_elbow, target_x, target_y]
        high_obs = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.render_mode = render_mode
        self.state: np.ndarray | None = None
        self.steps_beyond_terminated = None

        # Initialize angular velocities
        self.omega_shoulder = 0  # rad/s
        self.omega_elbow = 0     # rad/s

        self.armconfig = Armconfig(SIZE_HUMERUS=200, WIDTH_HUMERUS=20, SIZE_RADIUS=300, WIDTH_RADIUS=10)
        self.armrenderer = None

        self.target_angle_deg = None
        self.theta_1_c = 70  # Initial shoulder angle (degrees)
        self.theta_2_c = 0   # Initial elbow angle (degrees)
        self.target_x  = 0
        self.target_y = 0

        # Muscle activations and names
        self.muscle_activations = np.zeros(6, dtype=np.float32)
        self.muscle_names = ['shoulder_flexor', 'shoulder_extensor', 
                             'elbow_flexor', 'elbow_extensor', 
                             'biarticular_1', 'biarticular_2']

        # Reward parameters
        self.epsilon_target = 100
        self.target_radius = 30
        self.eph = None

    from time import time
    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.eph.nb_step_done += 1
        self.eph.past_theta_1_values.append(self.theta_1_c)
        self.eph.past_theta_2_values.append(self.theta_2_c) 

        # Update muscle activations
        self.muscle_activations = np.clip(action, 0, 1)

        # Calculate joint torques
        shoulder_flexor = self.muscle_activations[0]
        shoulder_extensor = self.muscle_activations[1]
        elbow_flexor = self.muscle_activations[2]
        elbow_extensor = self.muscle_activations[3]

        tau_shoulder = (shoulder_flexor - shoulder_extensor) * self.F_max * self.r_shoulder
        tau_elbow = (elbow_flexor - elbow_extensor) * self.F_max * self.r_elbow

        # Compute angular accelerations with damping
        alpha_shoulder = (tau_shoulder - self.b * self.omega_shoulder) / self.I_shoulder
        alpha_elbow = (tau_elbow - self.b * self.omega_elbow) / self.I_elbow

        # Update angular velocities (Euler integration)
        self.omega_shoulder += alpha_shoulder * self.dt
        self.omega_elbow += alpha_elbow * self.dt

        # Update joint angles (convert rad/s to degrees change)
        delta_theta_1 = math.degrees(self.omega_shoulder * self.dt)
        delta_theta_2 = math.degrees(self.omega_elbow * self.dt)
        self.theta_1_c += delta_theta_1
        self.theta_2_c += delta_theta_2

        # Clamp angles to biomechanical limits
        self.theta_1_c = np.clip(self.theta_1_c, 0, 180)
        self.theta_2_c = np.clip(self.theta_2_c, 0, 150)

        # Determine current target
        if self.eph.nb_step_done > len(self.eph.target_cartesian):
            terminated = True
            target_x, target_y = self.eph.target_cartesian[-1]
        else:
            terminated = False
            target_x, target_y = self.eph.target_cartesian[self.eph.nb_step_done - 1]

        self.target_x = target_x
        self.target_y = target_y

        # Update the state vector to include angles, angular velocities, current accelerations, and target position
        self.state = np.array([
            self.theta_1_c,
            self.theta_2_c,
            self.omega_shoulder,
            self.omega_elbow,
            alpha_shoulder,
            alpha_elbow,
            self.target_x,
            self.target_y
        ], dtype=np.float32)

        # Calculate hand coordinates from the most recent joint angles
        hand_y = (self.armconfig.SIZE_HUMERUS * math.sin(degrees_to_radians(self.eph.past_theta_1_values[-1])) +
                  self.armconfig.SIZE_RADIUS * math.sin(degrees_to_radians(self.eph.past_theta_1_values[-1] + self.eph.past_theta_2_values[-1])))
        hand_x = (self.armconfig.SIZE_HUMERUS * math.cos(degrees_to_radians(self.eph.past_theta_1_values[-1])) +
                  self.armconfig.SIZE_RADIUS * math.cos(degrees_to_radians(self.eph.past_theta_1_values[-1] + self.eph.past_theta_2_values[-1])))

        # Compute reward based on the distance between the hand and target
        reward = -1 * np.sqrt(np.abs(hand_x - target_x)** 2 + np.abs(hand_y - target_y)** 2) 

        # reward = -1 * np.sqrt(np.abs(hand_x - target_x)** 2 + np.abs(hand_y - target_y)** 2)  -  self.eph.nb_step_done

        self.eph.current_reward = reward
        self.eph.cum_reward_episode += reward
        self.eph.past_action = action

        # Check termination condition
        if hand_x == target_x and hand_y == target_y:
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        low, high = 0, 180
        theta_init = [70, 0]  # Initial shoulder and elbow angles
        self.theta_1_c = theta_init[0]
        self.theta_2_c = theta_init[1]
        
        # Reset angular velocities
        self.omega_shoulder = 0
        self.omega_elbow = 0
        self.steps_beyond_terminated = None

        if FIXED_TARGET:
            # Define target intervals with different radius scales
            intervals = [
                (0, 500, 1),
                (500, 1200, 0.5),
                (1200, 1800, 0.25),
                (1800, MAX_EPISODE_STEPS, 0.125)
            ]
            
            self.target_cartesian = []
            self.target_angle_deg = []
            
            for start, end, scale in intervals:
                radius = self.armconfig.SIZE_HUMERUS + (self.armconfig.SIZE_RADIUS * scale)
                target = sample_point_on_circle(radius)
                target = np.around(target, 2).tolist()
                target_angle_rad = math.atan2(target[1], target[0])
                target_angle = math.degrees(target_angle_rad)
                target_angle = np.clip(target_angle, low, high)
                num_steps = end - start
                self.target_cartesian.extend([target] * num_steps)
                self.target_angle_deg.extend([target_angle] * num_steps)
                
            self.target_cartesian = self.target_cartesian[:MAX_EPISODE_STEPS]
            self.target_angle_deg = self.target_angle_deg[:MAX_EPISODE_STEPS]
        else:
            sampling_indices = np.linspace(0, MAX_EPISODE_STEPS - 1, 10, dtype=int)
            targets_cartesian = [sample_point_on_circle(self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS) for _ in sampling_indices]
            target_angle_deg = []
            for t in targets_cartesian:
                target_angle_rad = math.atan2(t[1], t[0])
                target_angle_deg.append(math.degrees(target_angle_rad))
            spline = CubicSpline(sampling_indices, target_angle_deg)
            x_interpolated = np.linspace(0, MAX_EPISODE_STEPS, MAX_EPISODE_STEPS)
            self.target_angle_deg = spline(x_interpolated)
            self.target_angle_deg = np.clip(self.target_angle_deg, low, high)
            self.target_angle_deg = [round(t, 2) for t in self.target_angle_deg]
            self.target_cartesian = []
            for t in self.target_angle_deg:
                radius = self.armconfig.SIZE_HUMERUS + self.armconfig.SIZE_RADIUS
                x = radius * np.cos(degrees_to_radians(t))
                y = radius * np.sin(degrees_to_radians(t))
                self.target_cartesian.append(np.around([x, y], 2).tolist())

        self.eph = EpisodeHistory(MAX_EPISODE_STEPS,
                                  self.epsilon_target, 
                                  self.target_radius, 
                                  self.target_angle_deg, 
                                  self.target_cartesian)

        self.target_x = self.eph.target_cartesian[0][0]
        self.target_y = self.eph.target_cartesian[0][1]
        # Initialize state: set angular velocities and accelerations to 0
        self.state = np.array([
            self.theta_1_c,
            self.theta_2_c,
            self.omega_shoulder,
            self.omega_elbow,
            0.0,    # initial alpha_shoulder
            0.0,    # initial alpha_elbow
            self.target_x,
            self.target_y
        ], dtype=np.float32)

        if self.render_mode == "human":
            self.armrenderer = ArmRenderer(self.metadata, self.armconfig, self.eph)
            self.render()

        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, e.g. gym.make(\"{self.spec.id}\", render_mode=\"rgb_array\")"
            )
            return
        self.armrenderer.render()

    def close(self):
        if self.render_mode == 'human':
            self.armrenderer.close()



# For A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
def main():
    env = ArmReachingEnv2DTheta(render_mode=None)
    env = DummyVecEnv([lambda: env])
    
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=600000)
    model.save("td3_arm_reaching_A2C")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(f"Reward: {rewards}")
        # print(rewards)
        # env.envs[0].render()

    # env.close()

if __name__ == "__main__":
    main()







# With PPO
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # Create the environment; use None for render_mode during training
    env = ArmReachingEnv2DTheta(render_mode=None)
    env = DummyVecEnv([lambda: env])
    
    # Instantiate the TD3 model with the "MlpPolicy"
    model = TD3("MlpPolicy", env, verbose=1)
    
    # Train the model
    model.learn(total_timesteps=600000)
    
    # Save the trained model
    model.save("td3_arm_reaching_TD")

    # To evaluate or run the trained model, reset the environment
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(f"Reward: {rewards}")
        # Render the environment (if your environment supports rendering)
        # env.envs[0].render()

    # env.close()

if __name__ == "__main__":
    main()
