'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        shape of inputs:
        u_t0: [3]
        u_t1: [3]
        x_t0: [batch_size, 3]
        x_t1: [batch_size, 3]
        """

        """
        TODO : Add your code here
        """
        
        u_t0, u_t1 = np.array(u_t0), np.array(u_t1)

        # edge case: if odometry readings are exactly the same, return the previous state
        if u_t0 == u_t1:
            return x_t0

        # now we assume that there is motion
        # decompose method into three sections: rotation, translation, rotation
        # (0) find difference between u_t0 and u_t1 to make things more legible
        du = u_t1 - u_t0
        dx, dy, dtheta = du

        # (1) calculate first rotation
        motion_direction = np.arctan2(dy, dx)       # in range [-np.pi, np.pi]
        dtheta_1 = motion_direction - u_t0[2]

        # calculate translation
