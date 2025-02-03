#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 


#note this is now in 2D.

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize
import openmm

import numpy.typing as nptypes

from MSM import *

import config

def gaussian_2d_cov(x: nptypes.NDArray, y: nptypes.NDArray, gaussian_param: nptypes.NDArray) -> nptypes.NDArray:
    amplitude = gaussian_param[0]
    mean = gaussian_param[1:3].copy()
    covariance = gaussian_param[3:].reshape(2, 2)

    inv_covariance = np.linalg.inv(covariance)
    
    X_diff = x - mean[0]
    Y_diff = y - mean[1]
    
    # Compute the quadratic form of the Gaussian exponent for each x, y pair
    exponent = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            diff = np.array([X_diff[i, j], Y_diff[i, j]])
            exponent[i, j] = np.dot(diff.T, np.dot(inv_covariance, diff))
    
    # Finally, compute the Gaussian
    return amplitude * np.exp(-0.5 * exponent)

def gaussian_2d(x: nptypes.NDArray, y: nptypes.NDArray, ax: nptypes.NDArray, bx: nptypes.NDArray, by: nptypes.NDArray, cx: nptypes.NDArray, cy: nptypes.NDArray) -> nptypes.NDArray:
    return ax * np.exp(-((x - bx) ** 2 / (2 * cx ** 2) + (y - by) ** 2/(2 * cy ** 2)))


def count_transitions(b: np.int64, numbins: int, lagtime: int, endpt: np.int64 = None) -> Tuple[nptypes.NDArray, nptypes.NDArray]:
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape = (b.shape[0], numbins, numbins), dtype = np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k,  endpt[k, i], b[k, i - lagtime]] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym

    return sumtr.real, trvec


class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    numbins = config.num_bins
    lagtime = 1

    def __init__(self, gaussian_params: nptypes.NDArray, X: nptypes.NDArray):
        #gaussian_params comes in shape [prop_index + 1, num_gaussian, 5]
        num_gaussian = gaussian_params.shape[1]
        self.gaussian_params = gaussian_params

        self.x, self.y = np.meshgrid(X[0], X[1])
        self.X = X
        self.N = config.num_bins
        return

    def setup(self, CV: nptypes.NDArray, T: int, prop_index: int):
        self.data = CV
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index
        return

    def build_MM(self, sumtr: nptypes.NDArray, trvec: nptypes.NDArray, biased = False):
        N = self.numbins
        MM = np.empty(shape = (N * N, N * N), dtype = np.longdouble)
        if biased:
            MM = np.zeros(shape = (N * N, N * N), dtype = np.longdouble)
            for i in range(N * N):
                for j in range(N * N):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        i_x, i_y = np.unravel_index(i, (self.numbins, self.numbins), order = 'C')
                        j_x, j_y = np.unravel_index(j, (self.numbins, self.numbins), order = 'C')

                        for k in range(trvec.shape[0]):
                            #compute the total bias u.
                            u = np.zeros_like(self.x)
                            for g in range(self.gaussian_params.shape[1]):
                                if self.gaussian_params.shape[2] == 7:
                                    gaussian_param_slice = self.gaussian_params[k, g,:]
                                    u += gaussian_2d_cov(self.x, self.y, gaussian_param_slice)
                                elif self.gaussian_params.shape[2] == 5:    
                                    ax, bx, by, cx, cy = self.gaussian_params[k, g,:]
                                    u += gaussian_2d(self.x, self.y, ax, bx, by, cx, cy)
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp((u[j_x, j_y] - u[i_x, i_y]) / (2 * self.KbT))
                        if sump1 > 0:
                            MM[i, j] = sumtr[i, j] / sump1
                        else:
                            MM[i, j] = 0

            for i in range(MM.shape[0]):
                if np.sum(MM[i,:]) > 0:
                    MM[i,:] = MM[i,:] / np.sum(MM[i,:])
                else:
                    MM[i,:] = 0
        else:
            raise NotImplementedError("Not biased is not implemented!!!")
        
        return MM

    def run(self, biased = False):

        #digitialize the data into 2D mesh.
        b = self.data.astype(np.int64)
        
        sumtr, trvec = count_transitions(b, self.numbins * self.numbins, self.lagtime)

        MM = self.build_MM(sumtr, trvec, biased)

        msm = MSM()
        msm.build_MSM_from_M(MM, dim = 2, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
        msm.qspace = np.stack((self.x, self.y), axis = 1)
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M()
        msm._kemeny_constant_check()

        print("sum of peq from DHAM", np.sum(msm.peq))

        return msm.free_energy, msm.M
