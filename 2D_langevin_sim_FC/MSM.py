#this is the Class for the MSM
#by Tiejun Wei 13th Dec 2023

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import eig, logm
from scipy.linalg import inv
from scipy.constants import k as kB
import time
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
#from util import get_total_bias_2d
import config

class MSM:
    def __init__(self,time_step = 0.1,kbt = 0.5981,verbose = False,record_time = True):
        self.record_time = record_time
        self.K = None
        self.peq = None
        self.M = None
        self.M_unbiased = None
        self.time_step = time_step
        self.mfpts = None
        self.free_energy = None
        self.num_states = None
        self.kBT = kbt
        self.num_dimensions = None
        self.qspace = None #the phase space that we are interested in.
        self.verbose = verbose
        self.time_log = {}

    def update_log(self,name,s,e):
        if self.record_time:
            try:
                self.time_log[name].append(e - s)
            except:
                self.time_log[name] = [e - s]
        else:
            return

    def build_MSM_from_M(self, M, dim, time_step):
        """
        M is the transition matrix
        dim is the dimension of the phase space
        """
        s = time.time()
        self.M = M
        self.num_dimensions = dim
        self.num_states = int(np.round(M.shape[0] ** (1/self.num_dimensions)))
        self.time_step = time_step
        e = time.time()
        self.update_log('build_MSM_from_M',s,e)
    
    def build_MSM_from_data(self, data, lagtime):
        return
    
    def _compute_peq_fes_K(self):
        """
        K is the rate matrix
        kT is the thermal energy
        peq is the stationary distribution #note this was defined as pi in Simian's code.
        F is the free energy
        eigenvectors are the eigenvectors of K

        first we calculate the eigenvalues and eigenvectors of K
        then we use the eigenvalues to calculate the equilibrium distribution: peq.
        then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
        """
        s = time.time()
        evalues, evectors = eig(self.K)
        index = np.argsort(evalues)
        evalues_sorted = evalues[index]
        self.peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]])
        self.free_energy = -self.kBT * np.log(self.peq)
        e = time.time()
        self.update_log('_compute_peq_fes_K',s,e)
    
    def _compute_peq_fes_M(self):
        """
        M is the transition matrix
        kT is the thermal energy
        peq is the stationary distribution #note this was defined as pi in Simian's code.
        F is the free energy
        eigenvectors are the eigenvectors of K

        first we calculate the eigenvalues and eigenvectors of K
        then we use the eigenvalues to calculate the equilibrium distribution: peq.
        then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
        """
        s = time.time()
        evalues, evectors = eig(self.M.T)
        index = np.argsort(evalues)
        evalues_sorted = evalues[index]
        self.peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]).real
        self.free_energy = -self.kBT * np.log(self.peq)
        e = time.time()
        self.update_log('compute_peq_fes_M',s,e)
    
    def _init_sample_K_(self, dim, plot_init_fes = False,max_matrix_size = 100,coeffs = [(1,4,0),(2,1,1),(3,3,0)]): #unfinished
        """
        initialise a sample K to test methods. Works for any number of dimensions.
        """
        self.num_dimensions = dim
        #self.num_states = np.emath.logn(dim,max_matrix_size)
        self.num_states = round(max_matrix_size**(1/dim))
        actual_size = self.num_states**self.nm_dimensions
        self.K = np.zeros((actual_size,actual_size))

        #create discretised free energy surface
        qs= []
        for i in range(dim):
            qs.append(np.linspace(0,2*np.pi,self.num_states))
        self.qspace = np.meshgrid(*qs)
        trigs = (np.sin,np.cos)
        A = 0
        for i in range(len(self.qspace)):
            A += coeffs[i][0] * trigs[coeffs[i][2]](coeffs[i][1] *self.qspace[i] )
        if plot_init_fes:
            plt.figure
            plt.contourf(self.qspace[0],self.qspace[1],A - A.min())
            plt.colorbar()
            plt.savefig(f'./init_fes_dim{dim}.png')
        
        A = A.ravel('C')
        for i in range(self.num_states**self.num_dimensions):
            for j in range(i, self.num_states**self.num_dimensions):
                coor_i, coor_j, is_adjacent = self._is_adjacent(i,j)
                if is_adjacent:
                    u_ij = A[j] - A[i]
                    self.K[i,j] = np.exp((u_ij/(2 * self.kBT)))
                    self.K[j,i] = np.exp((-u_ij/(2 * self.kBT)))
            self.K[i,i] = -np.sum(self.K[:,i])


    def _init_sample_K(self, dim, plot_init_fes=False):
        """
        initialize a sample K to test methods
        note this is 1D case.
        """
        self.num_dimensions = dim
        if self.num_dimensions == 1:
            self.num_states = 100
            self.K = np.zeros((self.num_states**self.num_dimensions, self.num_states**self.num_dimensions), dtype=np.float64)

            #we create a discretized fes
            x = np.linspace(0, 2*np.pi, self.num_states)
            self.qspace = x
            y1 = np.sin((x-np.pi))
            y2 = np.sin((x-np.pi)/2)
            amplitude = 10
            xtilt = 0.5
            y = (xtilt*y1 + (1-xtilt)*y2) * amplitude 

            if plot_init_fes:
                plt.figure
                plt.plot(x,(y-y.min()))
                plt.savefig(f'./init_fes_dim{dim}.png')
                plt.close()

            #we create a transition matrix based on the discretized fes
            """for i in range(self.num_states-1):
                self.K[i, i + 1] = np.exp((y[i+1] - y[i]) / (2 * self.kBT))
                self.K[i + 1, i] = np.exp((y[i] - y[i+1]) / (2 * self.kBT))
            for i in range(self.num_states):
                self.K[i, i] = 0
                self.K[i, i] = -np.sum(self.K[:, i])"""
            for i in range(self.num_states**self.num_dimensions):
                for j in range(i, self.num_states**self.num_dimensions):
                    coor_i, coor_j, is_adjacent = self._is_adjacent(i,j)
                    if is_adjacent:
                        u_ij = y[j] - y[i]
                        self.K[i,j] = np.exp((u_ij / (2 * self.kBT)))
                        self.K[j,i] = np.exp((-u_ij / (2 * self.kBT)))
                self.K[i,i] = -np.sum(self.K[:,i])

        elif self.num_dimensions == 2:
            self.num_states = 20
            self.K = np.zeros((self.num_states**self.num_dimensions, self.num_states**self.num_dimensions), dtype=np.float64)

            #init a 2D fes.
            x,y = np.linspace(-2, 2, self.num_states), np.linspace(-2, 2, self.num_states)
            X,Y = np.meshgrid(x,y)
            self.qspace = [X,Y]

            Z = np.sin(X) + np.cos(Y)
            if plot_init_fes:
                plt.figure
                plt.contourf(X,Y,(Z-Z.min()))
                plt.colorbar()
                plt.savefig(f'./init_fes_dim{dim}.png')
                plt.close()

            #create K.
            Z = Z.ravel(order='C')
            for i in range(self.num_states**self.num_dimensions):
                for j in range(i, self.num_states**self.num_dimensions):
                    coor_i, coor_j, is_adjacent = self._is_adjacent(i,j)
                    if is_adjacent:
                        u_ij = Z[j] - Z[i]
                        self.K[i,j] = np.exp((u_ij / (2 * self.kBT)))
                        self.K[j,i] = np.exp((-u_ij / (2 * self.kBT)))
                self.K[i,i] = -np.sum(self.K[:,i])
        
        elif self.num_dimensions == 3:
            self.num_states = 10
            self.K = np.zeros((self.num_states**self.num_dimensions, self.num_states**self.num_dimensions), dtype=np.float64)

            #init a 3D fes.
            x,y,z = np.linspace(-2, 2, self.num_states), np.linspace(-2, 2, self.num_states), np.linspace(-2, 2, self.num_states)
            X,Y,Z = np.meshgrid(x,y,z)
            self.qspace = [X,Y,Z]

            W = np.sin(X*4) + 2*np.cos(Y) + 3*np.sin(Z*3)  ### energy function
            if plot_init_fes:
                #in 3D, we use 3D axis with ax.plot_surface to plot the fes.

                #we slice the W into different levels.
                levels = np.linspace(W.min(), W.max(), 10)

                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                if True:
                    slice_index = self.num_states // 2  # Mid slice for example
                    ax.plot_surface(X[:,:,slice_index], Y[:,:,slice_index], W[:,:,slice_index], cmap = "coolwarm")
                elif False:
                    scatter = ax.scatter(X, Y, Z, c=W, cmap = "coolwarm")
                    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.savefig(f'./init_fes_dim{dim}.png')
                plt.close()

            #create K.
            W = W.ravel(order='C')
            for i in range(self.num_states**self.num_dimensions):
                for j in range(i, self.num_states**self.num_dimensions):
                    coor_i, coor_j, is_adjacent = self._is_adjacent(i,j)
                    if is_adjacent:
                        u_ij = W[j] - W[i]
                        self.K[i,j] = np.exp((u_ij / (2 * self.kBT)))
                        self.K[j,i] = np.exp((-u_ij / (2 * self.kBT)))
                self.K[i,i] = -np.sum(self.K[:,i])

    def _build_mfpt_matrix_K(self, check=False):
        """
        we build mfpt matrix from K
        """
        s = time.time()
        onevec = np.ones((self.num_states**self.num_dimensions, 1))
        Qinv = np.linalg.inv(self.peq.T @ onevec - self.K.T)
        mfpts = np.zeros((self.num_states**self.num_dimensions, self.num_states**self.num_dimensions))
        for j in range(self.num_states**self.num_dimensions):
            for i in range(self.num_states**self.num_dimensions):
                if self.peq[j] != 0:
                    mfpts[i, j] = 1 / self.peq[j] * (Qinv[j,j] - Qinv[i,j])
        self.mfpts = mfpts
        e = time.time()
        self.update_log('_build_mfpt_matrix_K',s,e)
        if check:
            self._kemeny_constant_check()

    def _build_MSM_from_K(self):
        """
        we build a MSM from K
        this is done using discrete time approximation via scipy expm.
        """
        s = time.time()
        self.M = expm(self.K*self.time_step)
        self.update_log('_build_MM_from_K',s,time.time())

    def _build_mfpt_matrix_M(self, check=False, method='diag'):
        """
        peq is the stationary distribution
        M is the transition matrix
        method ['diag', 'JJhunter']
        """
        if method == 'diag':
            s = time.time()
            # onevec = np.ones((self.num_states**self.num_dimensions, 1))
            onevec = np.ones((self.M.shape[0],1))
            
            I = np.diag(onevec[:,0])
            A = (self.peq.reshape(-1,1)) @ onevec.T
            A = A.T
            # print(self.num_states,self.num_dimensions)
            # print(A.shape, I.shape, self.M.T.shape)

            #### Should use M instead of M transpose, Feifan May 31st
            try:
                Qinv = inv(A + I - self.M)
            except:
                Qinv = np.linalg.pinv((A + I - self.M).astype(float))
            mfpts = np.zeros((self.M.shape[0], self.M.shape[1]))
            
            # if self.M.shape[0] != self.num_states**self.num_dimensions:
            #     print("Error ", self.M.shape[0], " is not equal to ", self.num_states**self.num_dimensions)
            # mfpts = np.zeros((self.num_states**self.num_dimensions, self.num_states**self.num_dimensions))
            for j in range(self.M.shape[0]):
                for i in range(self.M.shape[1]):
                    term1 = Qinv[j,j] - Qinv[i,j] + I[i,j]
                    if self.peq[j] *term1 == 0:
                        mfpts[i, j] = 1e12
                    else:
                        mfpts[i, j] = 1 / self.peq[j] * term1

            #now we account for self.time_step
            mfpts = mfpts / self.time_step
            self.update_log('diag_method',s,time.time())
        
        if method == 'JJhunter':
            from numba import njit, prange
            @njit()
            def jjhelper(m,PPD,SD,rD):
                for n in range(m-1, 0, -1):
                    #print(n,PPD[n, :n])
                    SD[n] = np.sum(PPD[n, :n])
                    #print(SD)
                    for i in range(n):
                        for j in range(n):
                            PPD[i,j] += PPD[i, n] * PPD[n, j] / SD[n]
                rD[0] = 1
                for n in range(1, m):
                    for i in range(n):
                        rD[n] += rD[i] * PPD[i, n] / SD[n]
                return PPD,rD

            def jjhunter_numba(P):
                P = np.array(P, dtype=np.float64)  # Using double precision
                m = len(P)
                I = np.eye(m, dtype=np.float64)
                e1 = np.ones((m, 1), dtype=np.float64)
                E = np.ones((m, m), dtype=np.float64)
                PPD = P.copy()
                rD = np.zeros(m, dtype=np.float64)
            
                PPD, rD = jjhelper(m,PPD,rD.copy(),rD)
                TOTD = np.sum(rD)
                pit_1D = rD / TOTD
                PiD = e1 @ pit_1D.reshape(1, -1)
                ZD = np.linalg.inv(I - P + PiD)
                DD = np.diag(1/np.diag(PiD))
                MD = (I - ZD + E * np.diag(ZD)) @ DD
                return MD
            s = time.time()
            mfpts = jjhunter_numba(self.M)
            self.update_log('jjhunter',s,time.time())
            mfpts = mfpts / self.time_step
        self.mfpts = mfpts
        if check:
            self._kemeny_constant_check()

    def _kemeny_constant_check(self):
        """
        if max!=min, then assert error
        """
        s = time.time()
        kemeny_array = np.zeros((int(self.M.shape[0]), 1))
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[0]):
                kemeny_array[i] += self.mfpts[i, j] * float(self.peq[j])
        if np.max(kemeny_array) - np.min(kemeny_array) > 1e-6:
            print("max and min of kemeny are: ", np.max(kemeny_array), np.min(kemeny_array))
            #raise ValueError("Kemeny constant check failed!")
        else:
            if self.verbose:
                print("Kemeny constant check passed!")
        self.update_log('kemeny_constant_check',s,time.time())
    
    def _plot_fes_(self,filename="./free_energy.png"):
        assert self.free_energy is not None, "free energy is not calculated yet!"
        assert self.qspace is not None, "qspace is not initialized yet!"
        if self.num_dimensions == 1:
            "in 1D we simply plot it"
            plt.figure()           
            plt.plot(self.qspace, (self.free_energy - self.free_energy.min()))
            plt.savefig(filename)
            plt.close()
        else:
            state_tuple = tuple([self.num_states for i in range(self.num_dimensions)])
            fes_unravel = np.reshape(self.free_energy - self.free_energy.min(),state_tuple ,order = 'C')




    def _plot_fes(self, filename="./free_energy.png"):
        """
        we simply plot the fes on the x axis on self.qspace
        """
        assert self.free_energy is not None, "free energy is not calculated yet!"
        assert self.qspace is not None, "qspace is not initialized yet!"
        
        if self.num_dimensions == 1:
            """
            in 1D we simply plot it.
                """
            plt.figure()
            plt.plot(self.qspace, (self.free_energy - self.free_energy.min()))
            plt.savefig(filename)
            plt.close()
        
        elif self.num_dimensions == 2:
            """
            in 2D we plot it as a contour plot.
            qspace should be a meshgrid-like data with shape [2, self.num_states, self.num_states]
                e.g. X,Y = np.meshgrid(x,y) then qspace = np.array([X,Y])
            free_energy should be in shape [self.num_states, self.num_states]
            """
            fes_unravel = np.reshape((self.free_energy-self.free_energy.min()), (self.num_states, self.num_states), order='C')

            plt.figure()
            plt.contourf(self.qspace[0], self.qspace[1], fes_unravel)
            plt.colorbar()
            plt.savefig(filename)
            plt.close()
        
        elif self.num_dimensions == 3:
            """
            In 3D we plot it as a surface plot.
            qspace should be a meshgrid-like data with shape [3, self.num_states, self.num_states, self.num_states]
                e.g. X,Y,Z = np.meshgrid(x,y,z) then qspace = np.array([X,Y,Z])
            free_energy should be in shape [self.num_states, self.num_states, self.num_states]
            """
            fes_unravel = np.reshape((self.free_energy-self.free_energy.min()), (self.num_states, self.num_states, self.num_states), order='C')

            levels = np.linspace(fes_unravel.min(), fes_unravel.max(), 10)

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            if True:
                slice_index = self.num_states // 2  # Mid slice for example
                ax.plot_surface(self.qspace[0][:,:,slice_index], self.qspace[1][:,:,slice_index], fes_unravel[:,:,slice_index], cmap = "coolwarm")
            elif False:
                scatter = ax.scatter(self.qspace[0], self.qspace[1], self.qspace[2], c=fes_unravel, cmap = "coolwarm")
                fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(filename)
            plt.close()

    def _bias_K(self, bias):
        """
        bias is in shape (num_states*num_dimensions) (ravelled from a multi-dimensional matrix with shape [a,b,c] to shape [a*b*c, 1])
        the K should be in shape (num_states**num_dimensions, num_states**num_dimensions)
        """
        #if bias is not 1D, we ravel it.
        s = time.time()
        if len(bias.shape) != 1:
            bias = bias.ravel(order='C')
        #first we assert the num_states and num_dimensions
        #print(bias.shape[0],self.num_dimensions,self.num_states)
        assert round(bias.shape[0] ** (1/self.num_dimensions)) == self.num_states, "bias shape is not compatible with num_states and num_dimensions!"
        assert self.K.shape[0] == self.K.shape[1], "K is not a square matrix!"
        assert round(self.K.shape[0] ** (1/self.num_dimensions)) == self.num_states, "K is not compatible with num_states and num_dimensions!" 

        K_biased = np.zeros_like(self.K)
        for i in range(self.num_states**self.num_dimensions):
            for j in range(i, self.num_states**self.num_dimensions):
                coor_i, coor_j, is_adjacent = self._is_adjacent(i,j)
                if is_adjacent:
                    u_ij = bias[j] - bias[i]
                    K_biased[i, j] = self.K[i, j] * np.exp(u_ij / (2 * self.kBT))
                    K_biased[j, i] = self.K[j, i] * np.exp(-u_ij / (2 * self.kBT))
        
            K_biased[i, i] = -np.sum(K_biased[:, i])
        
        self.K = K_biased.real
        self.update_log('bias_K',s,time.time())
        if self.verbose:
            print("K is biased!")
    
    def _bias_M(self, bias, method = 'logm_K'):
        """
        bias is in shape (num_states*num_dimensions) (ravelled from a multi-dimensional matrix with shape [a,b,c] to shape [a*b*c, 1])
        the M should be in shape (num_states**num_dimensions, num_states**num_dimensions)
        """
        s = time.time()
        #if bias is not 1D, we ravel it.
        if len(bias.shape) != 1:
            bias = bias.ravel(order='C')
        #first we assert the num_states and num_dimensions
        
        assert round(bias.shape[0] ** (1/self.num_dimensions)) == self.num_states, "bias shape is not compatible with num_states and num_dimensions!"
        assert self.M.shape[0] == self.M.shape[1], "M is not a square matrix!"
        assert round(self.M.shape[0] ** (1/self.num_dimensions)) == self.num_states, "M is not compatible with num_states and num_dimensions!" 

        if method == 'logm_K':
            #if no K avaliable, we build it from M
            if self.K is None:
                print("K is not avaliable, we build it from M via logm")
                self.K = logm(self.M) / self.time_step
            self._bias_K(bias)
            M_biased = expm(self.K*self.time_step)
        elif method == 'direct_bias':
            M_biased = np.zeros_like(self.M)
            for i in range(self.M.shape[0]):
                for j in range(i, self.M.shape[0]):
                    coor_i, coor_j, is_adjacent = self._is_adjacent(i,j, is_MM=True)
                    if is_adjacent:
                        u_ij = bias[j] - bias[i]

                        if u_ij > config.cutoff:
                            u_ij = config.cutoff
                        elif u_ij < - config.cutoff:
                            u_ij = - config.cutoff

                        M_biased[i, j] = self.M[i, j] * np.exp(u_ij / (2 * self.kBT))
                        M_biased[j, i] = self.M[j, i] * np.exp(-u_ij / (2 * self.kBT))
                
            for i in range(self.M.shape[0]):
                M_biased[i, i] = self.M[i, i]
            #normalization on col.
            # note, in the Matlab this is normalization on row, but in python we do it on col.
            for i in range(self.M.shape[0]):
                col_sum = np.sum(M_biased[i])
                if col_sum > 0:
                    M_biased[i] = M_biased[i] / col_sum
                else:
                    M_biased[i] = 0
        self.M_unbiased = self.M
        self.M = M_biased
        if self.verbose:
            print("M is biased!")
        self.update_log('bias_M_' + method,s,time.time())

    def _is_adjacent(self, i, j, is_MM=False):
        """
        i and j are two states
        we check if they are adjacent
        note the K/M matrix is in shape: (num_states**num_dimensions, num_states**num_dimensions)
        """
        """if i == j:
            return 1,1, False
        else:
            return 1,1, True"""
        coor_i = np.unravel_index(i, (self.num_states,)*self.num_dimensions, order='C') #e.g. when dim = 1, unravel to shape (100, 1) when dim = 2, unravel to shape (100, 100) when dim = 3, unravel to shape (100, 100, 100)
        coor_j = np.unravel_index(j, (self.num_states,)*self.num_dimensions, order='C')
        distance = np.linalg.norm(np.array(coor_i) - np.array(coor_j))
        if is_MM:
            #for MM, we return True no matter the distance.
            return coor_i, coor_j, True
        else:
            if distance == 1:
                if self.verbose:
                    #print("state {} and state {} are adjacent!".format(i, j))
                    print("coor {} and coor {} are adjacent!".format(coor_i, coor_j))
                return coor_i, coor_j, True
            else:
                return coor_i, coor_j, False

# if __name__ == "__main__":
#     msm = MSM()
#     dim = 3
#     #test dim=1 cases
#     if dim == 1:
#         msm._init_sample_K(dim=1, plot_init_fes=True)
#         msm._compute_peq_fes_K()
#         msm._plot_fes(filename = './free_energy_K.png')
#         msm._build_MSM_from_K()
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_M.png')

#         #msm._build_mfpt_matrix_K(check=True)
#         msm._build_mfpt_matrix_M(check=True, method='diag')
#         print(msm.mfpts)
#         msm._build_mfpt_matrix_M(check=True, method='JJhunter')
#         print(msm.mfpts)
#         print("MSM test passed!")
        
#         #or we can use DHAM to pass the M and num_state
#         #msm.M = np.array([[0.9, 0.1], [0.1, 0.9]])
#         #msm.num_states = 2

#         #here we test the bias_M
#         def gaussian(x, a, b, c):
#             return a * np.exp(-(x - b)**2 / 2 / c**2)
        
#         a = 5
#         b = 2
#         c = 1
#         bias = gaussian(msm.qspace, a, b, c)
#         #msm._bias_K(bias)
#         #msm._compute_peq_fes_K()
#         msm.K = None #we took K away.
#         msm._bias_M(bias, method='direct_bias')
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_Mbiased.png')

#     #here we test dim=2 cases.
#     if dim == 2:
#         msm._init_sample_K(dim=2, plot_init_fes=True)
#         msm._compute_peq_fes_K()
#         msm._plot_fes(filename = './free_energy_K_2D.png')
#         msm._build_MSM_from_K()
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_M_2D.png')

#         #msm._build_mfpt_matrix_K(check=True)
#         msm._build_mfpt_matrix_M(check=True, method='diag')
#         print(msm.mfpts)
#         msm._build_mfpt_matrix_M(check=True, method='JJhunter')
#         print(msm.mfpts)
#         print("MSM test passed!")
        
#         #here we test bias_M
#         def gaussian_2D(x, y, a, bx, by, cx, cy):
#             return a * np.exp(-((x - bx)**2 / 2 / cx**2 + (y - by)**2 / 2 / cy**2))
        
#         a = 2
#         bx = -1
#         by = 1
#         cx = 1
#         cy = 0.5

#         bias = gaussian_2D(msm.qspace[0], msm.qspace[1], a, bx, by, cx, cy)
#         msm._bias_M(bias, method='direct_bias')
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_Mbiased_2D.png')
#         print("MSM test passed!")

#     #to do: test the dim=3 cases.
#     if dim == 3:
#         msm._init_sample_K(dim=3, plot_init_fes=True)
#         msm._compute_peq_fes_K()
#         msm._plot_fes(filename = './free_energy_K_3D.png')
#         msm._build_MSM_from_K()
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_M_3D.png')

#         msm._build_mfpt_matrix_K(check=True)
#         msm._build_mfpt_matrix_M(check=True)

#         #here we test bias_M
#         def gaussian_3D(x, y, z, a, bx, by, bz, cx, cy, cz):
#             return a * np.exp(-((x - bx)**2 / 2 / cx**2 + (y - by)**2 / 2 / cy**2 + (z - bz)**2 / 2 / cz**2))
        
#         a = 2
#         bx = -1
#         by = 1
#         bz = 1
#         cx = 1
#         cy = 0.5
#         cz = 0.5

#         bias = gaussian_3D(msm.qspace[0], msm.qspace[1], msm.qspace[2], a, bx, by, bz, cx, cy, cz)
#         msm.K = None
#         msm._bias_M(bias)
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = './free_energy_Mbiased_3D.png')
#         print("MSM test passed!")
#     else:
#         msm._init_sample_K(dim = dim,plot_init_fes = True)
#         msm._compute_peq_fes_K()
#         msm._plot_fes(filename = f'./free_energy_K_{dim}D.png')
#         msm._build_MSM_from_K()
#         msm._compute_peq_fes_M()
#         msm._plot_fes(filename = f'./free_energy_M_{dim}D.png')

###########################################################################     Recycle Bin ##########################################################################################
        
# def try_and_optim_M(self,working_indices, N=20, num_gaussian=10, start_index=0, end_index=0, plot = False):
    #     """
    #     here we try different gaussian params 1000 times
    #     and use the best one (lowest mfpt) to local optimise the gaussian_params

    #     returns the best gaussian params

    #     input:
    #     M: the working transition matrix, square matrix.
    #     working_indices: the indices of the working states.
    #     num_gaussian: number of gaussian functions to use.
    #     start_state: the starting state. note this has to be converted into the index space.
    #     end_state: the ending state. note this has to be converted into the index space.
    #     index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    #     """
    #     #here we find the index of working_indices.
    #     # e.g. the starting index in the working_indices is working_indices[start_state_working_index]
    #     # and the end is working_indices[end_state_working_index]

    #     start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    #     end_state_working_index = np.argmin(np.abs(working_indices - end_index))

    #     start_state_working_index_xy = np.unravel_index(working_indices[start_state_working_index], (N, N), order='C')
    #     end_state_working_index_xy = np.unravel_index(working_indices[end_state_working_index], (N, N), order='C')
    #     print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)

    #     #now our M/working_indices could be incontinues. #N = M.shape[0]
    #     x,y = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)) #hard coded here. we need to change this.
    #     best_mfpt = 1e20 #initialise the best mfpt np.inf

    #     #here we find the x,y maximum and minimun in xy coordinate space, with those working index
    #     #we use this to generate the random gaussian params.
    #     working_indices_xy = np.unravel_index(working_indices, (N, N), order='C')

    #     for try_num in range(1000):
    #         if try_num > 0:
    #             self.M = self.M_unbiased
    #         rng = np.random.default_rng()
    #         a = np.ones(num_gaussian)
    #         bx = rng.uniform(0, 2*np.pi, num_gaussian)
    #         by = rng.uniform(0, 2*np.pi, num_gaussian)
    #         cx = rng.uniform(0.3, 1.5, num_gaussian)
    #         cy = rng.uniform(0.3, 1.5, num_gaussian)
    #         gaussian_params = np.concatenate((a, bx, by, cx, cy))

    #         total_bias = get_total_bias_2d(x,y, gaussian_params)

    #         #we truncate the total_bias to the working index.
    #         working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.
    #         self._bias_M(working_bias)
    #         #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
    #         self._compute_peq_fes_M()
    #         #print(peq)
    #         #print(sum(peq))

    #         self._build_mfpt_matrix_M()

    #         mfpt_biased = self.mfpts[start_state_working_index, end_state_working_index]

    #         if try_num % 100 == 0:
    #             self._kemeny_constant_check()
    #             print("random try:", try_num, "mfpt:", mfpt_biased)
    #         if best_mfpt > mfpt_biased:
    #             best_mfpt = mfpt_biased
    #             best_params = gaussian_params

    #     print("best mfpt:", best_mfpt)

    #     #now we use the best params to local optimise the gaussian params

    #     def mfpt_helper(gaussian_params, M, start_state_working_index = start_state_working_index, end_state_working_index = end_state_working_index, kT=0.5981, working_indices=working_indices):
    #         #print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)
    #         total_bias = get_total_bias_2d(x,y, gaussian_params)

    #         #we truncate the total_bias to the working index.
    #         working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

    #         self._bias_M(working_bias)
    #         #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
    #         self._compute_peq_fes_M()
    #         self._build_mfpt_matrix_M()
    #         self.M = self.M_unbiased
    #         mfpt_biased = self.mfpts[start_state_working_index, end_state_working_index]
    #         return mfpt_biased

    #     res = minimize(mfpt_helper, 
    #                     best_params, 
    #                     args=(self.M,
    #                             start_state_working_index, 
    #                             end_state_working_index,
    #                             working_indices), 
    #                     #method='Nelder-Mead',
    #                     method="L-BFGS-B", 
    #                     bounds= [(0.1, 2)]*num_gaussian + [(0, 2*np.pi)]*num_gaussian + [(0, 2*np.pi)]*num_gaussian + [(0.3, 1.5)]*num_gaussian + [(0.3, 1.5)]*num_gaussian,
    #                     tol=1e-4)

    #     #print("local optimisation result:", res.x)
    #     return res.x