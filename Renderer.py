import matplotlib.pyplot as plt 
import numpy as np
from Utils import *
import math
from gymnasium.error import DependencyNotInstalled
try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
    ) from e

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)


class ArmRenderer: 
    def __init__(self, metadata, armconfig, episode_history) -> None:
        
        self.metadata = metadata
        self.armconfig = armconfig
        self.eph = episode_history
        
        self.WIDTH = 500 * 2
        self.HEIGHT = 500 * 2.5

        # rendering        
        self.screen = None
        self.clock = None
        self.isopen = True

        self.img_humerus_orig = None
        self.img_radius_orig = None
        self.my_font = None


        
    def render(self):

        if self.eph.nb_step_done == 0:
            return
        
        if self.screen is None:
            pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
            self.my_font = pygame.font.SysFont('dejavusans', 30)
            pygame.init()

            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )
            
            self.img_humerus_orig = pygame.Surface((self.armconfig.WIDTH_HUMERUS, self.armconfig.SIZE_HUMERUS))  
            self.img_radius_orig = pygame.Surface((self.armconfig.WIDTH_RADIUS, self.armconfig.SIZE_RADIUS))  
            
            self.img_humerus_orig.set_colorkey(BLACK)  
            self.img_humerus_orig.fill(GREEN)  

            self.img_radius_orig.set_colorkey(BLACK)  
            self.img_radius_orig.fill(GREEN)  


        if self.clock is None:
            self.clock = pygame.time.Clock()

        
        
        self.screen.fill(WHITE)

        x_0 = self.WIDTH / 2
        y_0 = self.HEIGHT / 2

        theta_1 = self.eph.past_theta_1_values[-1] + 90
        theta_2 = self.eph.past_theta_1_values[-1] + (self.eph.past_theta_2_values[-1] / 2) + 90

        rot_humerus_deg = theta_1 
        rot_radius_deg = theta_2 

        rot_humerus_rad = degrees_to_radians(rot_humerus_deg)
        rot_radius_rad =  degrees_to_radians(rot_radius_deg) 

        # rotating the original image  
        new_img_humerus = pygame.transform.rotate(self.img_humerus_orig, rot_humerus_deg)  
        new_img_radius  = pygame.transform.rotate(self.img_radius_orig, rot_radius_deg)  

        rect_humerus = new_img_humerus.get_rect() 
        rect_humerus.center = self.get_humerus_center(x_0, y_0, rot_humerus_rad)

        rect_radius = new_img_radius.get_rect()  
        rect_radius.center = self.get_radius_center(x_0, y_0, rot_humerus_rad, rot_radius_rad)

        pygame.draw.circle(self.screen, RED, (x_0, y_0), 20) # shoulder

        pygame.draw.line(self.screen, BLACK, (0, y_0), (x_0*2,y_0), 3)
        pygame.draw.line(self.screen, BLACK, (x_0, 0), (x_0 , y_0*2), 3)

        elbow_center = self.get_humerus_point(x_0, y_0, self.armconfig.SIZE_HUMERUS,rot_humerus_rad)

        pygame.draw.circle(self.screen, RED, elbow_center, 20) # elbow

        hand = self.get_radius_point(x_0, y_0, self.armconfig.SIZE_RADIUS, rot_humerus_rad, rot_radius_rad)

        pygame.draw.circle(self.screen, RED, hand, 20) # hand

        hand_x, hand_y = get_hand_xy_coord(
            self.armconfig,
            self.eph.past_theta_1_values[-1],
            self.eph.past_theta_2_values[-1]
        )

        hand_x_display = x_0 + hand_x
        hand_y_display = y_0 - hand_y 

        target_color = RED
        reward_color = GRAY

        success = False
        if is_point_in_circle(
            np.array([hand_x, hand_y]), 
            self.eph.target_cartesian[self.eph.nb_step_done-1], 
            self.eph.target_radius):
            
            target_color = GREEN
            success = True

        target_x_display = x_0 + self.eph.target_cartesian[self.eph.nb_step_done-1][0]
        target_y_display = y_0 - self.eph.target_cartesian[self.eph.nb_step_done-1][1]
        pygame.draw.circle(self.screen, target_color, (target_x_display, target_y_display), self.eph.target_radius, width=2) # target
        pygame.draw.circle(self.screen, reward_color, (target_x_display, target_y_display), self.eph.epsilon_target, width=2) # target
        

        self.screen.blit(new_img_humerus, rect_humerus)  
        self.screen.blit(new_img_radius, rect_radius)

        text_surface_1 = self.my_font.render(f'[{self.eph.nb_step_done}] reward : {self.eph.current_reward:.3f} cum.reward : {self.eph.cum_reward_episode:.0f}', False, BLACK)
        text_surface_2 = self.my_font.render(f'SHOULDER: {rot_humerus_deg:.2f} --- ELBOW: {rot_radius_deg:.2f}', False, BLACK)
        
        self.screen.blit(text_surface_1, (10,10))
        self.screen.blit(text_surface_2, (10,50))
        
        #self.draw_muscles(x_0, y_0, rot_humerus_rad, rot_radius_rad)

        
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()  

    def get_hand_xy_coord(self, armconfig, theta_1, theta_2):
        return (
            armconfig.SIZE_HUMERUS * math.sin(theta_1) + armconfig.SIZE_RADIUS * math.sin(theta_1 + theta_2),
            armconfig.SIZE_HUMERUS * math.cos(theta_1) + armconfig.SIZE_RADIUS * math.cos(theta_1 + theta_2)
        )    
    def is_point_in_circle(self, point, center, radius):
        return np.linalg.norm(point - center) < radius
    
    def degrees_to_radians(self, degrees):
        return math.radians(degrees)
    
    def get_humerus_point(self, x_0, y_0, point_loc, rot_rad):
        return (
            x_0  + (point_loc * math.sin(rot_rad)), 
            y_0  + (point_loc * math.cos(rot_rad))
        )

    def get_humerus_center(self,x_0, y_0, rot_rad):
        return self.get_humerus_point(x_0, y_0, self.armconfig.SIZE_HUMERUS/2, rot_rad)
    
    def get_radius_point(self, x_0, y_0, point_loc, rot_hum_rad, rot_radius_rad):
        base = self.get_humerus_point(x_0, y_0, self.armconfig.SIZE_HUMERUS, rot_hum_rad)
        return (
            base[0] + (point_loc * math.sin(rot_radius_rad)), 
            base[1] + (point_loc * math.cos(rot_radius_rad)) 
        )
    def get_radius_center(self,x_0, y_0, rot_hum_rad, rot_radius_rad):
        return self.get_radius_point(x_0, y_0, self.armconfig.SIZE_RADIUS/2, rot_hum_rad, rot_radius_rad)


    def get_radius_point(self, x_0, y_0, point_loc, rot_hum_rad, rot_radius_rad):
        base = self.get_humerus_point(x_0, y_0, self.armconfig.SIZE_HUMERUS, rot_hum_rad)
        return (
            base[0] + (point_loc * math.sin(rot_radius_rad)), 
            base[1] + (point_loc * math.cos(rot_radius_rad)) 
        )
    def get_radius_center(self,x_0, y_0, rot_hum_rad, rot_radius_rad):
        return self.get_radius_point(x_0, y_0, self.armconfig.SIZE_RADIUS/2, rot_hum_rad, rot_radius_rad)
    

    def apply_target_driven_contraction(self, hand_x, hand_y, target_x, target_y):
        distance_to_target = math.dist((hand_x, hand_y), (target_x, target_y))
        contraction_factor = 1 - min(distance_to_target / 500, 1)  # Closer = more contraction
        return contraction_factor


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False






