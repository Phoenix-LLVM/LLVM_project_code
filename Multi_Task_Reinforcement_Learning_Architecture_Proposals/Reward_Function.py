

import numpy as np
import carla 
import time  # to set a delay after each photo
import cv2   # to work with images from cameras
import numpy as np # in this example for reshaping
import json
import random
import utils
import math
max_distance    = 3.0 
target_speed    = 28.0 

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])
    
def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def reward_kendall(vehicle_speed):
    speed_kmh = round(3.6 * math.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2),0)
    return speed_kmh

def reward_speed_centering_angle_add(vehicle_velocity, current_waypoint, next_waypoint,vehicle_location):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 27.0 # km/h
    max_speed = 30.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(vehicle_velocity)
    wp_fwd = vector(current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = round(3.6 * math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2),0)
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]
    
    distance_from_center = distance_to_line(vector(current_waypoint.transform.location),
                                                     vector(next_waypoint.transform.location),
                                                     vector(vehicle_location))
    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward + centering_factor + angle_factor

    return reward
