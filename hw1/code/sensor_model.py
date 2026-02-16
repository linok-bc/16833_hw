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
    def __init__(self, map_reader, subsampling=10, batch_size=10):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self.map_reader = map_reader
        # self._z_hit = 1
        # self._z_short = 0.1
        # self._z_max = 0.1
        # self._z_rand = 100
        self._z_hit   = 0.80
        self._z_short = 0.10
        self._z_max   = 0.05
        self._z_rand  = 0.05

        # self._sigma_hit = 50
        # self._lambda_short = 0.1
        self._sigma_hit = 60.0      # cm (start 50–100)
        self._lambda_short = 0.01   # 1/cm  (mean 100 cm)

        # Used in p_max and p_rand, optionally in ray casting
        # For this project, we seem to be using cm; so this will be 100 grid units assuming that it's isotropic w/ 10cm res.
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = subsampling

        # Used for processing the data in batches; adjust based on memory contraints (especially with these RAM prices)
        self._batch_size = batch_size
       
    def ray_casting(self, x_t1):
        """
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]; has shape [self._batch_size, 3]
        param[in] map_reader : representation of the environment using the MapReader class
        param[out] out : estimated distance (in # of cm) before the robot hits an obstacle; shape [self._batch_size, 180//self._subsampling]
        
        got rid of map_reader param cuz we're passing into constructor
        """

        """
        explanation of variables

        batch_size: number of particles being processed at a single time
        num_samples: number of ray directions being considered, calculated as 180 // self._subsampling
        max_range_units: the max range of the sensors expressed in map units rather than cm
        """

        # by default, coordinates are stored in absolute terms (i.e. cm) rather than map units
        max_range_units = np.ceil(self._max_range / self.map_reader._resolution).astype(np.uintp)

        # for each particle, convert to (subsampled) rays in the hemisphere of the robot's direction
        x_t1_cone = np.repeat(np.expand_dims(x_t1, 1), 180 // self._subsampling, 1)
        x_t1_cone[:, :, 2] -= np.array(range(-90, 90, self._subsampling)) * (np.pi / 180)   # [batch_size, num_samples, 3]

        # currently we have [x, y, theta]; let's convert that to [x, y, dx, dy] through sin/cos
        x_t1_cone = np.dstack([x_t1_cone, x_t1_cone[:, :, 2]])
        x_t1_cone[:, :, 2] = np.cos(x_t1_cone[:, :, 2])
        x_t1_cone[:, :, 3] = np.sin(x_t1_cone[:, :, 3])                                     # [batch_size, num_ray_samples, 4]
        
        # unfortunately we need to sample along each direction a lot; RAM goes whee
        x_t1_cone = np.repeat(np.expand_dims(x_t1_cone, 2), max_range_units, axis=2)        # [batch_size, num_samples, max_range_units, 4]
        x_t1_cone[:, :, :, 2:] *= (np.array(range(max_range_units)) * self.map_reader._resolution)[np.newaxis, np.newaxis, :, np.newaxis] 

        # convert from position + offset to new position (e.g (x, dx) -> (x+dx))
        x_t1_cone = x_t1_cone[..., :2] + x_t1_cone[..., 2:]                                 # [batch_size, num_samples, max_range_units, 2]

        # convert resolution of these units from cm to map units, and snap to grid
        x_t1_cone /= self.map_reader._resolution
        x_t1_cone = np.round(x_t1_cone)

        # handles cases for when we leave the grid; clipping is equivalent to repeating the behavior at the edge of the grid
        x_t1_cone[..., 0] = np.clip(x_t1_cone[..., 0], 0, self.map_reader._occupancy_map.shape[0]-1)
        x_t1_cone[..., 1] = np.clip(x_t1_cone[..., 1], 0, self.map_reader._occupancy_map.shape[1]-1)
        x_t1_cone = x_t1_cone.astype(np.uintp)

        # convert from indices of grid to occupancy score 
        x_t1_cone = self.map_reader.get_map()[x_t1_cone[..., 0], x_t1_cone[..., 1]]              # [batch_size, num_samples, max_range_units]

        # threshold occupancy score to get a binary mask, then convert to distance to the nearest obstacle (or max)
        x_t1_cone = x_t1_cone >= self._min_probability
        x_t1_cone = np.where(                                                               # [batch_size, num_samples]
            x_t1_cone, 
            np.array(range(max_range_units))[np.newaxis, np.newaxis, :], 
            max_range_units
        ).min(axis=-1) 
        x_t1_cone *= self.map_reader._resolution         

        return x_t1_cone


    # def beam_range_finder_model(self, z_t1_arr, x_t1):
    #     """
    #     param[in] z_t1_arr : laser range readings [array of 180 values] at time t
    #     param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    #     param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        
        
        
    #     z_t1_arr is size 180 array of scalar laser range readings at time t
    #     x_t1 is single particle pose (x,y, heading)
    #     we want to return scalar likelihood we get range scan z_t1_arr given we're at position x_t1
        
    #     """
        
    #     """
    #     TODO : Add your code here
    #     """
    #     # prob_zt1 = 1.0
    #     # return prob_zt1
        
    #     eps = 1e-8 # use in log sum at end to avoid log(0)
    #     z_max = self._max_range
        
    #     # subsample so that measured scan matches scan from raycasting
    #     z = np.array(z_t1_arr, dtype=float)
    #     z = z[::self._subsampling]
        
    #     # clamp to make sure we're not over max
    #     z = np.clip(z, 0, z_max)
        
    #     # get predicted scan by raycasting for batch size 1 (single position)
    #     x = np.array(x_t1, dtype=float).reshape(1,3)
        
    #     z_star = self.ray_casting(x)[0]  # should be 1d
    #     z_star = np.clip(z_star, 0, z_max)
        
    #     # ok so now z and z star should be same length
    #     assert(len(z) == len(z_star))
    #     # now we can iterate through k, and calculate the mixture for each one
    #     likelihoods = []
    #     for z_k, z_k_star in zip(z, z_star):
        
    #         # 1. calc phit
    #         # gaussian pdf center zkstar with sigma hit, eval at zk
    #         phit = norm.pdf(z_k, loc=z_k_star, scale=self._sigma_hit) if (0 <= z_k <= z_max) else 0
    #         # 2. calc pshort
    #         pshort = expon.pdf(z_k, loc=0, scale=1/self._lambda_short) if (0 <= z_k <= z_k_star) else 0
    #         # 3. calc pmax
    #         pmax = 1 if (z_k == z_max) else 0
    #         # 4. calc prand
    #         # prand = 1/z_max if (0 <= z_k < z_max) else 0
    #         pmax = 1.0 if z_k >= z_max - 1e-3 else 0.0  # use small tolerance instead of exact equality
            
    #         # note coef self._z_max_ is different from max sensor range z_max
    #         mixture = self._z_hit * phit + self._z_short * pshort + self._z_max * pmax + self._z_rand * prand
            
    #         # likelihoods.append(mixture + eps)  # avoid log(0)
    #         likelihoods.append(mixture + eps)  # don't want 0 likelihood

        
    #     # return log likelihod
    #     # actually assignment says in practice product works better?
    #     return np.prod(likelihoods)
    #     # return np.sum(np.log(likelihoods))
    
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Batched sensor model.

        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle states [B,3] (or [3]) at time t [world_frame]
        param[out] prob_zt1 : likelihood per particle, shape [B]
        """

        eps = 1e-8
        z_max = float(self._max_range)

        # --- z: subsample once (shared across particles) ---
        z = np.asarray(z_t1_arr, dtype=np.float64)[::self._subsampling]
        z = np.clip(z, 0.0, z_max)                              # [K]
        K = z.shape[0]

        # --- x: ensure batched ---
        x = np.asarray(x_t1, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, 3)
        assert x.ndim == 2 and x.shape[1] == 3
        B = x.shape[0]

        # --- predicted ranges for all particles ---
        z_star = self.ray_casting(x)                            # [B, K]
        z_star = np.clip(z_star, 0.0, z_max)

        # --- vectorized mixture components over [B,K] ---
        # broadcast z to [B,K]
        z_bk = z[None, :]                                       # [1, K] -> broadcast to [B, K]
        z_bk = np.repeat(z_bk, z_star.shape[0] ,axis=0)

        # p_hit: Gaussian pdf N(z; z*, sigma) on [0, z_max]
        phit = norm.pdf(z_bk, loc=z_star, scale=self._sigma_hit)  # [B, K]
        phit *= ((z_bk >= 0.0) & (z_bk <= z_max))

        # p_short: exponential on [0, z*]
        # scipy expon.pdf(x, scale=1/lambda) == lambda * exp(-lambda x)
        pshort = expon.pdf(z_bk, loc=0.0, scale=1.0 / self._lambda_short)  # [B, K]
        pshort *= ((z_bk >= 0.0) & (z_bk <= z_star))

        # p_max: point mass at z_max (use tolerance)
        pmax = (z_bk >= (z_max - 1e-3)).astype(np.float64)       # [B, K]

        # p_rand: uniform on [0, z_max)
        prand = (1.0 / z_max) * ((z_bk >= 0.0) & (z_bk < z_max)) # [B, K]

        # mixture per beam
        mix = (
            self._z_hit   * phit +
            self._z_short * pshort +
            self._z_max   * pmax +
            self._z_rand  * prand
        )                                                        # [B, K]

        mix = mix + eps

        # --- return per-particle likelihood ---
        # product version (as assignment tip suggests); may underflow if K is large
        return np.prod(mix, axis=1)                               # [B]

        # If you instead want the numerically-stable log-likelihood, use:
        # return np.sum(np.log(mix), axis=1)     
                
            
            
    
    

