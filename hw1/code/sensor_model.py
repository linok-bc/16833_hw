'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm, expon

from map_reader import MapReader

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        # For this project, we seem to be using cm; so this will be 100 grid units assuming that it's isotropic w/ 10cm res.
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 10

        # Used for processing the data in batches; adjust based on memory contraints (especially with these RAM prices)
        self._batch_size = 10
       
    def ray_casting(self, x_t1, map_reader):
        """
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]; has shape [self._batch_size, 3]
        param[in] map_reader : representation of the environment using the MapReader class
        param[out] out : estimated distance (in # of cm) before the robot hits an obstacle; shape [self._batch_size, 180//self._subsampling]
        """

        # for ease, I'm going to convert the max range to map units here
        max_range_mapframe = np.ceil(self._max_range / map_reader._resolution)

        # for each particle, convert to (subsampled) rays in the hemisphere of the robot's direction
        x_t1_cone = np.repeat(np.expand_dims(x_t1, 1), 180 // self._subsampling, 1)
        x_t1_cone[:, :, 2] -= np.array(range(-90, 90, self._subsampling)) / (np.pi / 180)        # [batch_size, num_samples, 3]

        # currently we have [x, y, theta]; let's convert that to [x, y, dx, dy] through sin/cos
        x_t1_cone = np.dstack([x_t1_cone, x_t1_cone[:, :, 2]])
        x_t1_cone[:, :, 2] = np.cos(x_t1_cone[:, :, 2] * (np.pi/180))
        x_t1_cone[:, :, 3] = np.sin(x_t1_cone[:, :, 3] * (np.pi/180))           # [batch_size, num_samples, 4]
        
        # unfortunately we need to sample along each direction a lot; RAM goes whee
        x_t1_cone = np.repeat(np.expand_dims(x_t1_cone, 2), self._max_range)    # [batch_size, num_samples, max_rang_mapframe, 4]
        x_t1_cone[:, :, :, 2:] *= np.array(range(1, self.max_range))

        # 


        out = None
        return out


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]; has shape [num_particles, 3]
        param[in] step : step at which to draw samples from the beam_range finer
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t; should have shape [num_particles,]
        """
        """
        TODO : Add your code here
        """

        # only consider a portion of z_t1_arr to save on computation
        z_t1_arr_sample = z_t1_arr[::self._subsampling]

        # From chapter 6.3, we will consider four different types of noise and four corresponding distributions
        
        ##########
        # (1) local measurement noise, which is modelled with a Gaussian
        ##########
        
        # minimum # of std. dev.'s: z_t1 / sigma_hit
        # maximum # of std. dev.'s: (self._max_range - z_t1) / sigma_hit
        lower_cdf = norm.cdf(-(z_t1_arr_sample / self._sigma_hit))
        upper_cdf = norm.cdf((self._max_range - z_t1_arr_sample) / self._sigma_hit)
        
        # get noise samples in terms of cdf values, then convert to # of std. dev.'s
        noise_samples = np.random.uniform(lower_cdf, upper_cdf, size=z_t1_arr_sample.shape)
        noise_samples = norm.ppf(noise_samples, scale=self._sigma_hit)

        # add noise to z_t1 samples to get z_hit
        z_hit = z_t1_arr_sample + noise_samples

        ##########
        # (2) unexpected obstacles, which we model with an exponential
        ##########
        
        # minimum's c.d. is just at 0
        # maximum's c.d. is at c.d. of the measurement
        lower_cdf = expon.cdf(0)
        upper_cdf = expon.cdf(z_t1_arr_sample, scale=1/self._lambda_short)

        # get noise values in terms of c.d. values, then convert
        noise_samples = np.random.uniform(lower_cdf, upper_cdf, size=z_t1_arr_sample.shape)
        noise_samples = expon.ppf(noise_samples, scale=1/self._lambda_short)
        z_short = noise_samples

        ##########
        # (3) sensor failure, which we model with a point-mass distribution
        ##########
        z_max = np.repeat(np.array([[self._max_range]]), z_t1_arr_sample.shape, axis=0)

        ##########
        # (4) random measurement error, which we model with a uniform distribution
        ##########
        z_rand = np.random.uniform(low=0, high=self._max_range, size=z_t1_arr_sample.shape)

        # now we calculate weighed averages, using matrix multiplication to go wheee
        z_all = np.stack([z_hit, z_short, z_max, z_rand])
        weights = np.array([self._z_hit, self._z_short, self._z_max, self._z_rand])
        weights /= weights.sum()
        weighted_z_all = weights @ z_all


        prob_zt1 = 0
        return prob_zt1
