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

    def sample(self, d_rot1, d_trans, d_rot2, batch_size):
        """
        Sample noise to add to our odometry estimates. Each sample in x_t0 gets its own noise sample

        param[in] d_rot1 : estimated first rotation from odometry
        param[in] d_trans : estimated translation from odometry
        param[in] d_rot2 : estimated second rotation from odometry
        param[in] batch_size : first parameter of x_t* has shape batch_size
        param[out] x_noise: noises values with shape [batch_size, 3] and template [d_rot1, d_trans, d_rot2] along the last dimension
        """

        # (1) calculate standard deviations of noise Gaussians
        std_rot1 = self._alpha1 * d_rot1 ** 2 + self._alpha2 * d_trans ** 2
        std_trans = self._alpha3 * (d_rot1 ** 2 + d_rot2 ** 2) + self._alpha4 * d_trans ** 2
        std_rot2 = self._alpha1 * d_rot2 ** 2 + self._alpha2 * d_trans ** 2

        # (2) sample noise from distributions. We don't need to scale since \sigma already depends on the translation 
        noise_rot1 = np.random.normal(loc=0, scale=std_rot1, size=batch_size)
        noise_trans = np.random.normal(loc=0, scale=std_trans, size=batch_size)
        noise_rot2 = np.random.normal(loc=0, scale=std_rot2, size=batch_size)

        # (3) create and return x_noise
        x_noise = np.stack([noise_rot1, noise_trans, noise_rot2], axis=-1)
        return x_noise


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
        Thrun et al., Probabalistic Robotics, Table 5.6
        """
        
        u_t0, u_t1 = np.array(u_t0), np.array(u_t1)

        # edge case: if odometry readings are exactly the same, return the previous state
        if np.allclose(u_t0, u_t1):
            return x_t0

        # now we assume that there is motion
        # decompose method into three sections: rotation, translation, rotation
        # (0) find difference between u_t0 and u_t1 to make things more legible
        du = u_t1 - u_t0
        dx, dy, dtheta = du

        # (1) calculate rotations and translations
        d_rot1 = np.arctan2(dy, dx) - u_t0[2]
        d_trans = np.sqrt(dy**2 + dx**2)
        d_rot2 = dtheta - d_rot1

        # (2) add sampled noise to estimated rotations and translation
        x_noise = self.sample(d_rot1, d_trans, d_rot2, x_t0.shape[0])
        du_noisy = np.array([d_rot1, d_trans, d_rot2])[np.newaxis, :] + x_noise
        du_rot1, du_trans, du_rot2 = du_noisy[:, 0], du_noisy[:, 1], du_noisy[:, 2]

        # (3) convert noisy rotation and translations to changes in position and orientation
        du_x = du_trans * np.cos(x_t0[:, 2] + du_rot1)
        du_y = du_trans * np.sin(x_t0[:, 2] + du_rot1)
        du_theta = du_rot1 + du_rot2
        x_t1 = x_t0 + np.stack([du_x, du_y, du_theta], axis=-1)
        
        return x_t1
