import math
import numpy as np

WHITE = (255, 255, 255)  
BLACK = (0 , 0 , 0)  
GREEN = (0 , 255 , 0)  
RED = (255 , 0 , 0)  
GRAY = (128 , 128 , 128)  


class EpisodeHistory: 
    def __init__(self, max_episode_steps, epsilon_target, target_radius, target_angle_deg, target_cartesian) -> None:


        self.max_episode_steps = max_episode_steps
        self.epsilon_target = epsilon_target
        self.target_radius = target_radius
        self.target_angle_deg = target_angle_deg
        self.target_cartesian = target_cartesian
        self.nb_step_done = 0
        self.cum_reward_episode = 0
        self.current_reward = 0

        self.past_action = None
        self.past_delta_theta_1_control = [] # asked
        self.past_delta_theta_1_changed = [] # delta theta observed after the dynamics
        self.past_theta_1_values = []
        self.past_theta_2_values = []

        self.ep_speed_history = []
        self.ep_jerk_history = []


def degrees_to_radians(degrees):
    # Normalize the degrees to be between 0 and 360 (optional, but common)
    degrees = degrees % 360
    return degrees * (math.pi / 180)

def is_point_in_circle(point, center, radius):
    x_point, y_point = point[0], point[1]
    x_center, y_center = center[0], center[1]

    distance = math.sqrt((x_point - x_center) ** 2 + (y_point - y_center) ** 2)
    if distance <= radius:
        return True
    else:
        return False
    
def sample_point_on_circle(radius, only_upper_part=True):
    # Sample a random angle theta between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi)

    print(theta)
    
    # Calculate x and y coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    if only_upper_part and y < 0:
        y = -y
    
    return x, y

def moving_average_smoothing(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    # Pad to make the smoothed array the same length as the original
    pad_size = (len(data) - len(smoothed)) // 2
    smoothed = np.pad(smoothed, (pad_size, len(data) - len(smoothed) - pad_size), mode='edge')
    return smoothed

def get_hand_xy_coord(armconfig, theta_1_c, theta_2_c):
        
    return (
        armconfig.SIZE_HUMERUS * math.cos(degrees_to_radians(theta_1_c)) + armconfig.SIZE_RADIUS * math.cos(degrees_to_radians(theta_1_c)+degrees_to_radians(theta_2_c)),
        armconfig.SIZE_HUMERUS * math.sin(degrees_to_radians(theta_1_c)) + armconfig.SIZE_RADIUS * math.sin(degrees_to_radians(theta_1_c)+degrees_to_radians(theta_2_c))
    ) 