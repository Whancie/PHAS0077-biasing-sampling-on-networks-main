#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 


#note this is now in 2D.

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize
import openmm

import config

def gaussian_2d_cov(x, y, gaussian_param):
    amplitude = gaussian_param[0]
    mean = gaussian_param[1:3].copy()
    covariance = gaussian_param[3:].reshape(2,2)

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

def gaussian_2d(x, y, ax, bx, by, cx, cy):
    return ax * np.exp(-((x-bx)**2/(2*cx**2) + (y-by)**2/(2*cy**2)))

def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


def count_transitions(b, numbins, lagtime, endpt=None):
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                #Ntr[k, b[k, i - lagtime], endpt[k, i]] += 1
                Ntr[k,  endpt[k, i], b[k, i - lagtime]] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    #plt.contourf(sumtr.real)
    #plt.colorbar()
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

    def __init__(self, gaussian_params,X):
        #gaussian_params comes in shape [prop_index + 1, num_gaussian, 5]
        num_gaussian = gaussian_params.shape[1]
        self.gaussian_params = gaussian_params
        # x,y = np.meshgrid(np.linspace(0, 2*np.pi, self.numbins), np.linspace(0, 2*np.pi, self.numbins))
        self.x, self.y = np.meshgrid(X[0],X[1])
        self.X = X
        self.N = config.num_bins
        return

    def setup(self, CV, T, prop_index, time_tag):
        self.data = CV
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index
        self.time_tag = time_tag
        return

    def build_MM(self, sumtr, trvec, biased=False):
        N = self.numbins
        MM = np.empty(shape=(N*N, N*N), dtype=np.longdouble)
        if biased:
            MM = np.zeros(shape=(N*N, N*N), dtype=np.longdouble)
            for i in range(N*N):
                for j in range(N*N):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        i_x, i_y = np.unravel_index(i, (self.numbins, self.numbins), order='C')
                        j_x, j_y = np.unravel_index(j, (self.numbins, self.numbins), order='C')

                        for k in range(trvec.shape[0]):
                            #compute the total bias u.
                            u = np.zeros_like(self.x)
                            for g in range(self.gaussian_params.shape[1]):
                                if self.gaussian_params.shape[2] == 7:
                                    gaussian_param_slice = self.gaussian_params[k, g,:]
                                    u += gaussian_2d_cov(self.x, self.y, gaussian_param_slice)
                                elif self.gaussian_params.shape[2] == 5:    
                                    ax, bx, by, cx, cy = self.gaussian_params[k, g, :]
                                    u += gaussian_2d(self.x, self.y, ax, bx, by, cx, cy)
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp((u[j_x, j_y] - u[i_x, i_y]) / (2*self.KbT))
                        if sump1 > 0:
                            MM[i, j] = sumtr[i, j] / sump1
                        else:
                            MM[i, j] = 0
            #epsilon_offset = 1e-15
            #MM = MM / (np.sum(MM, axis=1)[:, None] + 1e-15) #normalize the M matrix #this is returning NaN?.
            for i in range(MM.shape[0]):
                if np.sum(MM[i, :]) > 0:
                    MM[i, :] = MM[i, :] / np.sum(MM[i, :])
                else:
                    MM[i, :] = 0
        else:
            raise NotImplementedError
        
        #plt.contourf(MM.real)
        return MM

    def run(self, plot=True, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        #digitialize the data into 2D mesh.
        #b = np.digitize(self.data, np.linspace(0, 2*np.pi, self.numbins*self.numbins+1))
        b = self.data.astype(np.int64)
        
        sumtr, trvec = count_transitions(b, self.numbins * self.numbins, self.lagtime)

        MM = self.build_MM(sumtr, trvec, biased)
        """#MM = MM.T  # to adapt our script pattern.
        d, v = eig(MM.T)
        mpeq = v[:, np.where(d == np.max(d))[0][0]]
        mpeq = mpeq / np.sum(mpeq)
        mpeq = mpeq.real
        #rate = np.float_(- self.lagtime * conversion / np.log(d[np.argsort(d)[-2]]))
        mU2 = - self.KbT * np.log(mpeq)
        #dG = np.max(mU2[:int(self.numbins)])
        #A = rate / np.exp(- dG / self.KbT)
        """
        #from util import compute_free_energy, compute_free_energy_power_method, Markov_mfpt_calc, kemeny_constant_check
        from MSM import MSM

        msm = MSM()
        msm.build_MSM_from_M(MM, dim = 2, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
        msm.qspace = np.stack((self.x,self.y), axis=1)
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M()
        msm._kemeny_constant_check()

        #print("peq", peq)
        print("sum of peq from DHAM", np.sum(msm.peq))

        if False:
            # plt.figure()
            # plt.imshow(np.reshape(mU2,[self.numbins, self.numbins], order = 'C'), cmap = 'coolwarm', origin='lower')
            # plt.savefig("./test.png")
            # plt.close()


            plt.figure()
            plt.plot(msm.peq)
            plt.savefig('peq_MM.png')

            plt.figure()
            plt.imshow(np.reshape(msm.peq, (20,20), order='C'), cmap='coolwarm', origin='lower')
            plt.colorbar()
            plt.savefig('peq_MM_2D.png')


        return msm.free_energy, msm.M
