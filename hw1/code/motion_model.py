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
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]; has shape [num_particles, 3]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]; has shape [num_particles, 3]
        """
        """
        TODO : Add your code here
        """

        # (1) estimate displacement in odometry frame
        du = np.array([u_t1[0] - u_t0[0], u_t1[1] - u_t0[1], u_t1[2] - u_t0[2]])

        # (2) expand to [num_particles, 3] and add some estimated noise
        du_expanded = np.expand_dims(0, du).repeat(x_t0.shape[0])
        du_noisy = du_expanded  # TODO

        # (3) utilize displacement in odometry frame to calculate displacement in world frame
        x_t1 = x_t0 + du_noisy
        return x_t1
