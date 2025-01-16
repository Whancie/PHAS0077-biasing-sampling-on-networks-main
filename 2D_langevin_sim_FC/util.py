import numpy as np
from MSM import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import openmm
import config
import pickle as pk
from sklearn.decomposition import PCA
from numba import prange, njit

@njit(parallel = True)
def gaussian_6d_mesh_correct(mesh, amplitude, mean, covariance):
    inv_covariance = np.linalg.inv(covariance)
    
    X1_diff = mesh[0] - mean[0]
    Y1_diff = mesh[1] - mean[1]
    Z1_diff = mesh[2] - mean[2]
    X2_diff = mesh[3] - mean[3]
    Y2_diff = mesh[4] - mean[4]
    Z2_diff = mesh[5] - mean[5]

    # Compute the quadratic form of the Gaussian exponent for each x, y pair
    exponent = np.zeros(mesh[0].shape)
    for i in prange(mesh[0].shape[0]):
        for j in prange(mesh[0].shape[1]):
            for k in prange(mesh[0].shape[2]):
                for l in prange(mesh[0].shape[3]):
                    for m in range(mesh[0].shape[4]):
                        for n in range(mesh[0].shape[5]):
                            diff = np.array([X1_diff[i, j, k, l, m , n], Y1_diff[i, j, k, l, m , n], Z1_diff[i, j, k, l, m , n],X2_diff[i, j, k, l, m , n], Y2_diff[i, j, k, l, m , n], Z2_diff[i, j, k, l, m , n]])
                            exponent[i, j, k, l, m, n] = np.dot(diff.T, np.dot(inv_covariance, diff))
                
    # Finally, compute the Gaussian
    return amplitude * np.exp(-0.5 * exponent)

# Updating the get_total_bias_6D function to use the corrected gaussian_6d_mesh function
def get_total_bias_6D(mesh, gaussian_params):
    N = mesh[0].shape[0]
    total_bias = np.zeros((N,) * 6)
    for params in gaussian_params:
        amplitude = params[0]
        mean = params[1:7]
        covariance = params[7:].reshape(6,6)
        total_bias += gaussian_6d_mesh_correct(mesh, amplitude, mean, covariance + 1e-6 * np.eye(6))
    return total_bias

def random_initial_bias_2d(initial_position, num_gaussians = 20):
    initial_position = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[0]

    rng = np.random.default_rng()
    a = np.ones(num_gaussians) * 0.01#* 4 #
    #ay = np.ones(num_gaussians) * 0.1 #there's only one amplitude!
    bx = rng.uniform(initial_position[0]-1, initial_position[0]+1, num_gaussians)
    by = rng.uniform(initial_position[1]-1, initial_position[1]+1, num_gaussians)
    cx = rng.uniform(1.0, 5.0, num_gaussians)
    cy = rng.uniform(1.0, 5.0, num_gaussians)
    #"gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."
    return np.concatenate((a, bx, by, cx, cy))


def random_initial_bias_Nd(initial_position, num_gaussians = 20, amplitude = 0.01, num_dimensions = 6):
    initial_position1 = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[0]
    initial_position2 = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[1]

    rng = np.random.default_rng()
    a = amplitude * np.ones((num_gaussians, 1))
    mean_vecs = []
    all_stdevs = []
    for i in range(num_dimensions//2):
        mean_vecs.append(rng.uniform(initial_position1[i]-1, initial_position1[i]+1, num_gaussians))
        all_stdevs.append(rng.uniform(1.0, 5.0, num_gaussians))
    for i in range(num_dimensions//2):
        mean_vecs.append(rng.uniform(initial_position2[i]-1, initial_position2[i]+1, num_gaussians))
        all_stdevs.append(rng.uniform(1.0, 5.0, num_gaussians))
    mean_vecs = np.array(mean_vecs).T
    all_stdevs = np.array(all_stdevs).T
    all_cov = []
    for i in range(len(all_stdevs)):
        all_cov.append(np.diag(all_stdevs[i]**2).flatten())
    all_cov = np.array(all_cov)
    return np.concatenate([a,mean_vecs, all_cov], axis = 1).T.flatten()


def pca_transform(data, pca_fitting_file_path = "./pca_coords/pca.pkl", mode = "to_pca"):
    # if mode == "create_pca":
    #     pca_ = PCA(n_components=2)
    #     principal_variance = pca_.fit(data)
    #     pk.dump(principal_variance, open(pca_fitting_file_path, "wb"))
    #     return principal_variance.transform(data)
    
    if mode == "to_pca":    
        pca_reload = pk.load(open(pca_fitting_file_path,"rb"))
        return pca_reload.transform(data)
    elif mode == "from_pca":
        pca_reload = pk.load(open(pca_fitting_file_path,"rb"))
        return pca_reload.inverse_transform(data)
    elif mode == "matrix_to_pca":
        pca_reload = pk.load(open(pca_fitting_file_path,"rb"))
        return np.dot(pca_reload.components_,np.dot(data, pca_reload.components_.T))
    elif mode == "matrix_from_pca":
        pca_reload = pk.load(open(pca_fitting_file_path,"rb"))
        return np.dot(pca_reload.components_.T,np.dot(data, pca_reload.components_))
    else:
        raise ValueError(f"Invalid Mode Detected: {mode}")

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 
def try_and_optim_M(M,X, working_indices, prop_index, N=20, num_gaussian=20, start_index=0, end_index=0, plot = False):
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params

    input:
    M: the working transition matrix, square matrix.
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    """
    #here we find the index of working_indices.
    # e.g. the starting index in the working_indices is working_indices[start_state_working_index]
    # and the end is working_indices[end_state_working_index]
    
    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    
    #### start_state_working_index_xy needs to be ravelled in F order
    start_state_working_index_xy = np.unravel_index(working_indices[start_state_working_index], (N, N), order='F')
    end_state_working_index_xy = np.unravel_index(working_indices[end_state_working_index], (N, N), order='C')
    print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)

    #now our M/working_indices could be incontinues. #N = M.shape[0]
    #x,y = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)) #hard coded here. we need to change this. ### qspace
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    #here we find the x,y maximum and minimun in xy coordinate space, with those working index
    #we use this to generate the random gaussian params.
    working_indices_xy = np.unravel_index(working_indices, (N, N), order='C')

    msm = MSM()
    msm.qspace = np.stack((X[0], X[1]),axis=1)
    #msm.build_MSM_from_M(M, dim = 2, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds)) #changed by TW 14th May
    msm.build_MSM_from_M(M, dim = 1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
    msm._compute_peq_fes_M()

    #parameter settings.
    pca_x_min = X[0][0]
    pca_x_max = X[0][-1]
    pca_y_min = X[1][0]
    pca_y_max = X[1][-1]

    c_low_factor = 0.1/(2*np.pi) #defines the factor used in definition of lower barrier of cx.
    c_high_factor = 1/(2*np.pi) 


    X_mesh, Y_mesh = np.meshgrid(X[0], X[1])
    
    for try_num in range(100):
       # msm.build_MSM_from_M(M, dim = 2, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds)) #changed by TW 14th May
        msm.build_MSM_from_M(M, dim = 1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
        rng = np.random.default_rng()
        a = np.ones(num_gaussian) * 1
        bx = rng.uniform(pca_x_min, pca_x_max, num_gaussian)
        by = rng.uniform(pca_y_min, pca_y_max, num_gaussian)
        cx = rng.uniform(c_low_factor * (pca_x_max - pca_x_min), c_high_factor * (pca_x_max - pca_x_min), num_gaussian) #for [0, 2pi], c is [0.1, 1]
        cy = rng.uniform(c_low_factor * (pca_y_max - pca_y_min), c_high_factor * (pca_y_max - pca_y_min), num_gaussian)
        gaussian_params = np.concatenate((a, bx, by, cx, cy))
        total_bias = get_total_bias_2d(X_mesh, Y_mesh, gaussian_params) #note this should take in the np.meshgrid object. NOT NP ARRAY!

        #we truncate the total_bias to the working index.
        working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

        #now we have a discontinues M matrix. we need to apply the bias to the working index.
        msm._bias_M(working_bias, method = "direct_bias")
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M(method = "diag")
        mfpts_biased = msm.mfpts
        
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        # print(mfpt_biased)

        # ### find the negative mfpt_biased
        # if mfpt_biased < 0:
        #     print(f"Negative mfpt_biased: {mfpt_biased}")
        #     print(f"Try number: {try_num}")
        #     print(f"Start state: {start_state_working_index}")
        #     print(f"End state: {end_state_working_index}")

        if try_num % 100 == 0:
            msm._kemeny_constant_check()
            print("random try:", try_num, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = gaussian_params

    print("best mfpt:", best_mfpt)
    
    best_bias = get_total_bias_2d(X_mesh, Y_mesh, best_params)
    #here we plot the total bias just to check the bias applied in pca space make sence.
    if True:


        plt.figure()
        plt.imshow(best_bias,cmap="coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin="lower")
        plt.title("best_bias in the pca space.")
        plt.colorbar()
        plt.savefig(f'./test_best_bias_p{prop_index}.png')
        plt.close()


    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(gaussian_params, M, start_state_working_index = start_state_working_index, end_state_working_index = end_state_working_index, kT=0.5981, working_indices=working_indices_xy):
        #print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)
        msm = MSM()
        msm.qspace = np.stack((X[0], X[1]),axis=1)
        #msm.build_MSM_from_M(M, dim = 2, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
        msm.build_MSM_from_M(M, dim = 1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
        total_bias = get_total_bias_2d(X_mesh, Y_mesh, gaussian_params)
        working_bias = total_bias[working_indices]
        msm._bias_M(working_bias, method = 'direct_bias')
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M(method = "diag")
        mfpts_biased = msm.mfpts
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]       
        #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
        return mfpt_biased


    ##### skip minimize to do a test

    res = minimize(mfpt_helper,
                   best_params, 
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices_xy), 
                   #method='Nelder-Mead',
                   method="L-BFGS-B", 
                   bounds= [(0.1, 1.1)]*num_gaussian + [(pca_x_min, pca_x_max)]*num_gaussian + [(pca_y_min, pca_y_max)]*num_gaussian + [(c_low_factor * (pca_x_max - pca_x_min),c_high_factor * (pca_x_max - pca_x_min))]*num_gaussian + [(c_low_factor* (pca_y_max - pca_y_min), c_high_factor * (pca_y_max - pca_y_min))]*num_gaussian,
                   tol=1e-4)
    
    #print("local optimisation result:", res.x)
    gaussian_param = res.x

    if True:
        optimised_bias = get_total_bias_2d(X_mesh, Y_mesh, gaussian_param)
        plt.figure()
        plt.imshow(optimised_bias,cmap="coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin="lower")
        plt.title("optimised_bias in the pca space.")
        plt.colorbar()
        plt.savefig(f'./test_optimised_bias_p{prop_index}.png')
        plt.close()



    # gaussian_param = best_params
    A = gaussian_param[:num_gaussian]
    x0 = gaussian_param[num_gaussian:2*num_gaussian]
    y0 = gaussian_param[2*num_gaussian:3*num_gaussian]
    sigma_x = gaussian_param[3*num_gaussian:4*num_gaussian]
    sigma_y = gaussian_param[4*num_gaussian:5*num_gaussian]

    mean_vec = []
    cov_matrix = []
    for i in range(num_gaussian):
        mean_vec.append(np.array([x0[i], y0[i]]))
        cov_matrix.append(np.array([[sigma_x[i]**2,0],[0,sigma_y[i]**2]]))

    # Also temporarily return best_params
    # return gaussian_param, np.array(mean_vec), np.array(cov_matrix)
    
    return res.x, np.array(mean_vec), np.array(cov_matrix)

def apply_fes(system, particle_idx, gaussian_param=None, pbc = False, name = "FES", amp = 7, mode = "gaussian", plot = False, plot_path = "./fes_visualization.png"):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    pi = np.pi #we need convert this into nm.
        #at last we add huge barrier at the edge of the box. since we are not using pbc.
    #this is to prevent the particle from escaping the box.
    # if x<0, push the atom back to x=0


    k = 5  # Steepness of the sigmoid curve
    max_barrier = "1e2"  # Scaling factor for the potential maximum
    offset = 0.7 #the offset of the boundary energy barrier.
    # Defining the potentials using a sigmoid function
    left_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * x - (-{offset}))))")
    right_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (x - (2 * {pi} + {offset})))))")
    bottom_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * y - (-{offset}))))")
    top_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (y - (2 * {pi} + {offset})))))")

    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    bottom_pot.addParticle(particle_idx)
    top_pot.addParticle(particle_idx)

    system.addForce(left_pot)
    system.addForce(right_pot)
    system.addForce(bottom_pot)
    system.addForce(top_pot)

    
    #unpack gaussian parameters
    if mode == "gaussian":
        num_gaussians = int(len(gaussian_param)/5)
        A = gaussian_param[0::5] * amp #*7
        x0 = gaussian_param[1::5]
        y0 = gaussian_param[2::5]
        sigma_x = gaussian_param[3::5]
        sigma_y = gaussian_param[4::5]

        #now we add the force for all gaussians.
        energy = "0"
        force = openmm.CustomExternalForce(energy)
        for i in range(num_gaussians):
            if pbc:
                energy = f"A{i}*exp(-periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) - periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)
            else:
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

            #examine the current energy term within force.

            print(force.getEnergyFunction())

            force.addGlobalParameter(f"A{i}", A[i])
            force.addGlobalParameter(f"x0{i}", x0[i])
            force.addGlobalParameter(f"y0{i}", y0[i])
            force.addGlobalParameter(f"sigma_x{i}", sigma_x[i])
            force.addGlobalParameter(f"sigma_y{i}", sigma_y[i])
            force.addParticle(particle_idx)
            #we append the force to the system.
            system.addForce(force)
        if plot:
            #plot the fes.
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(num_gaussians):
                Z += A[i] * np.exp(-(X-x0[i])**2/(2*sigma_x[i]**2) - (Y-y0[i])**2/(2*sigma_y[i]**2))
            plt.figure()
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
            plt.xlabel("x")
            plt.xlim([-1, 2*np.pi+1])
            plt.ylim([-1, 2*np.pi+1])
            plt.ylabel("y")
            plt.title("FES mode = gaussian, pbc=False")
            plt.colorbar()
            plt.savefig(plot_path)
            plt.close()
            fes = Z

    if mode == "multiwell":
        """
        here we create a multiple well potential.
         essentially we deduct multiple gaussians from a flat surface, 
         with a positive gaussian acting as an additional barrier.
         note we have to implement this into openmm CustomExternalForce.
            the x,y is [0, 2pi]
         eq:
            U(x,y) = amp * (1                                                                   #flat surface
                            - A_i*exp(-(x-x0i)^2/(2*sigma_xi^2) - (y-y0i)^2/(2*sigma_yi^2))) ...        #deduct gaussians
                            + A_j * exp(-(x-x0j)^2/(2*sigma_xj^2) - (y-y0j)^2/(2*sigma_yj^2))       #add a sharp positive gaussian
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_wells = 9
            num_barrier = 1

            #here's the well params
            A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
            x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
            y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
            sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
            sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

            #here's the barrier params
            # for example we define a diagonal barrier at x = pi
            A_j = np.array([0.3]) * amp
            x0_j = [np.pi]
            y0_j = [np.pi]
            sigma_x_j = [3]
            sigma_y_j = [0.3]

            #now we add the force for all gaussians.
            #note all energy is in Kj/mol unit.
            energy = str(amp * 4.184) #flat surface
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            for i in range(num_wells):
                energy = f"-A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i}", A_i[i] * 4.184) #convert kcal to kj
                force.addGlobalParameter(f"x0{i}", x0_i[i])
                force.addGlobalParameter(f"y0{i}", y0_i[i])
                force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
                force.addGlobalParameter(f"sigma_y{i}", sigma_y_i[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            for i in range(num_barrier):
                energy = f"A{i+num_wells}*exp(-(x-x0{i+num_wells})^2/(2*sigma_x{i+num_wells}^2) - (y-y0{i+num_wells})^2/(2*sigma_y{i+num_wells}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i+num_wells}", A_j[i])
                force.addGlobalParameter(f"x0{i+num_wells}", x0_j[i])
                force.addGlobalParameter(f"y0{i+num_wells}", y0_j[i])
                force.addGlobalParameter(f"sigma_x{i+num_wells}", sigma_x_j[i])
                force.addGlobalParameter(f"sigma_y{i+num_wells}", sigma_y_j[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            if plot:
                #plot the fes.
                x = np.linspace(0, 2*np.pi, 100)
                y = np.linspace(0, 2*np.pi, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                Z += amp * 4.184 #flat surface
                for i in range(num_wells):
                    Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
                for i in range(num_barrier):
                    Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))
                
                #add the x,y boundary energy barrier.
                total_energy_barrier = np.zeros_like(X)
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * pi + offset))))) #right
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * pi + offset)))))
                Z += total_energy_barrier

                plt.figure()
                plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp* 12/7 * 4.184, origin="lower")
                plt.xlabel("x")
                plt.xlim([-1, 1+2*np.pi])
                plt.ylim([-1, 1+2*np.pi])
                plt.ylabel("y")
                plt.title("FES mode = multiwell, pbc=False")
                plt.colorbar()
                plt.savefig(plot_path)
                plt.close()
                fes = Z
            
    if mode == "funnel":
        """
        this is funnel like potential.
        we start wtih a flat fes, then add/deduct sphrical gaussians
        eq:
            U = 0.7* amp * cos(2 * p * (sqrt((x-pi)^2 + (y-pi)^2))) #cos function. periodicity determines the num of waves.
            - amp exp(-((x-pi)^2+(y-pi)^2))
            + 0.4*amp*((x-pi/8)^2 + (y-pi/8)^2)
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for funnel potential.")
        else:
            plot_3d = False
            periodicity = 8
            energy = f"0.7*{amp} * cos({periodicity} * (sqrt((x-{pi})^2 + (y-{pi})^2))) - 0.6* {amp} * exp(-((x-{pi})^2+(y-{pi})^2)) + 0.4*{amp}*((x-{pi}/8)^2 + (y-{pi}/8)^2)"
            
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            if plot:
                if plot_3d:
                    import plotly.graph_objs as go

                    # Define the x, y, and z coordinates
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.9* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z -= 0.6* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2)/0.5)
                    Z += 0.4*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    #add the x,y boundary energy barrier.
                    total_energy_barrier = np.zeros_like(X)
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - 0)))) #left
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - 2 * pi)))) #right
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - 0))))
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - 2 * pi))))
                    Z += total_energy_barrier

                    # Create the 3D contour plot
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin = 0, cmax = amp *12/7)])
                    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
                    fig.update_layout(title='FES mode = funnel, pbc=False', autosize=True,
                                    width=800, height=800,
                                    scene = {
                                        "xaxis": {"nticks": 5},
                                        "yaxis": {"nticks": 5},
                                        "zaxis": {"nticks": 5},
                                        "camera_eye": {"x": 1, "y": 1, "z": 0.4},
                                        "aspectratio": {"x": 1, "y": 1, "z": 0.4}
                                    }
                                    )
                                    #margin=dict(l=65, r=50, b=65, t=90))
                    #save fig.
                    fig.write_image(plot_path)
                    fes = Z
                    
                else:
                    #plot the fes.
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.4* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z += 0.7* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2/0.5))
                    Z += 0.2*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    #add the x,y boundary energy barrier.
                    total_energy_barrier = np.zeros_like(X)
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - 0)))) #left
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - 2 * pi)))) #right
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - 0))))
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - 2 * pi))))
                    Z += total_energy_barrier

                    plt.figure()
                    plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
                    plt.xlabel("x")
                    plt.xlim([-1, 2*np.pi+1])
                    plt.ylim([-1, 2*np.pi+1])
                    plt.ylabel("y")
                    plt.title("FES mode = funnel, pbc=False")
                    plt.colorbar()
                    plt.savefig(plot_path)
                    plt.close()
                    fes = Z

    return system, fes #return the system and the fes (2D array for plotting.)


def apply_bias(system, particle_idx, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20):
    """
    this applies a bias using customexternal force class. similar as apply_fes.
    note this leaves a set of global parameters Ag, x0g, y0g, sigma_xg, sigma_yg.
    as these parameters can be called and updated later.
    note this is done while preparing the system before assembling the context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    num_gaussians = len(gaussian_param)//5
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"Ag{i}*exp(-(x-x0g{i})^2/(2*sigma_xg{i}^2) - (y-y0g{i})^2/(2*sigma_yg{i}^2))" #in openmm unit, kj/mol, nm.
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        force.addGlobalParameter(f"x0g{i}", x0[i]) #convert to nm
        force.addGlobalParameter(f"y0g{i}", y0[i])
        force.addGlobalParameter(f"sigma_xg{i}", sigma_x[i])
        force.addGlobalParameter(f"sigma_yg{i}", sigma_y[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)
    
    print("system added with bias.")
    return system


def update_bias(simulation, gaussian_param, name = "BIAS", num_gaussians = 20):
    """
    given the gaussian_param, update the bias
    note this requires the context object. or a simulation object.
    # the context object can be accessed by simulation.context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    num_gaussians = len(gaussian_param)//5
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]

    #now we update the GlobalParameter for all gaussians. with num_gaussians terms. and update them in the system.
    #note globalparameter does NOT need to be updated in the context.
    for i in range(num_gaussians):
        simulation.context.setParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        simulation.context.setParameter(f"x0g{i}", x0[i]) #convert to nm
        simulation.context.setParameter(f"y0g{i}", y0[i])
        simulation.context.setParameter(f"sigma_xg{i}", sigma_x[i])
        simulation.context.setParameter(f"sigma_yg{i}", sigma_y[i])
    
    print("system bias updated")
    return simulation

def gaussian_params_pca_real(means, standard_deviations, pca_fitting_file_path = "./pca_coords/pca.pkl"):
    assert len(means) == len(standard_deviations), "Error - Length of mean and standard deviation array not equal!"
    cov_matrix = []
    for i in range(len(means)):
        row = [0 for i in range(len(means))]
        row[i] = standard_deviations[i]**2
        cov_matrix.append[row]
    cov_matrix = np.array(cov_matrix)
    transformed_means = pca_transform(means.reshape(1,len(means)), pca_fitting_file_path = pca_fitting_file_path, mode = "from_pca")
    transformed_means = transformed_means.reshape(len(standard_deviations))
    transformed_cov_matrix = pca_transform(cov_matrix, pca_fitting_file_path = pca_fitting_file_path, mode = "matrix_to_pca")
    return transformed_cov_matrix, transformed_means


def gaussian_2D(x, y, a, bx, by, cx, cy):
    return a * np.exp(-(((x - bx)**2 / 2 / cx**2) + ((y - by)**2 / 2 / cy**2)))


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

def get_total_bias_2d(x,y, gaussian_params,no_covariance = True):
    """
    here we get the total bias at x,y.
    note: we used the transposed K matrix, we need to apply transposed total gaussian bias.
    """
    N = x.shape[0] #N is the number of grid points.
    total_bias = np.zeros((N,N))
    if no_covariance:
        num_gaussians = len(gaussian_params)//5
    
        a = gaussian_params[:num_gaussians]
        bx = gaussian_params[num_gaussians:2*num_gaussians]
        by = gaussian_params[2*num_gaussians:3*num_gaussians]
        cx = gaussian_params[3*num_gaussians:4*num_gaussians]
        cy = gaussian_params[4*num_gaussians:5*num_gaussians]

        for i in range(num_gaussians):
            total_bias = total_bias + gaussian_2D(x,y,a[i], bx[i], by[i], cx[i], cy[i])
    else:
        num_gaussians = gaussian_params.shape[0]
        for i in range(num_gaussians):
            total_bias += gaussian_2d_cov(x,y,gaussian_params[i])
    return total_bias



def apply_2D_bias_cov_FC(system, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20, epsilon = 0.1):
    assert len(gaussian_param) == 43 * num_gaussians

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians:(i+2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T
    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force1 = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        current_mean = means[i][:2]
        current_covs = covs[i].reshape(6,6)
        current_covs = current_covs[:2,:2]
        # inverted_cov = np.linalg.inv(current_covs)
        
        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy1 = f"A1g{i}*exp(-(x-x0g{i})^2 * var_y0_b{i}/(2*(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2)) - (y-y0g{i})^2 * var_x0_a{i}/(2*(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2)) + var_x0_b{i}*(x-x0g{i})*(y-y0g{i})/(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2))" #in openmm unit, kj/mol, nm.
            #energy1 = f"A1g{i}*exp(-(x-x0g{i})^2/(2*var_x0_a{i}) - (y-y0g{i})^2/(2*var_y0_b{i}) + var_x0_b{i}*(x-x0g{i})*(y-y0g{i})/sqrt(var_x0_a{i}*var_y0_b{i}))" #in openmm unit, kj/mol, nm.
            #energy1 = f"A1g{i}*exp(-(x-x0g{i})^2/(2*(var_x0_a{i} + var_x0_b{i})) - (y-y0g{i})^2/(2*(var_y0_a{i} + var_y0_b{i} )))" #in openmm unit, kj/mol, nm.
            
            force1 = openmm.CustomExternalForce(energy1)

            # energy2 = f"A2g{i}*exp(-(x-x1g{i})^2/(2*(var_x1_a{i} + var_x1_b{i} + var_x1_c{i} + var_x1_d{i} + var_x1_e{i} + var_x1_f{i})) - (y-y1g{i})^2/(2*(var_y1_a{i} + var_y1_b{i} + var_y1_c{i} + var_y1_d{i} + var_y1_e{i} + var_y1_f{i})) - (z-z1g{i})^2/(2*(var_z1_a{i} + var_z1_b{i} + var_z1_c{i} + var_z1_d{i} + var_z1_e{i} + var_z1_f{i})))" #in openmm unit, kj/mol, nm.
            # force2 = openmm.CustomExternalForce(energy2)

        #examine the current energy term within force.

        print(force1.getEnergyFunction())

        force1.addGlobalParameter(f"A1g{i}", A[i] * 4.184) #convert to kJ/mol
        force1.addGlobalParameter(f"x0g{i}", current_mean[0]) #convert to nm
        force1.addGlobalParameter(f"y0g{i}", current_mean[1])
        
        cov_row_label = ["x0","y0"]
        cov_col_label = ["a","b"]
        for j in range(2):
            for k in range(2):
                force1.addGlobalParameter(f"var_{cov_row_label[j]}_{cov_col_label[k]}{i}", current_covs[j,k])
        force1.addParticle(0)
        #we append the force to the system.
        system.addForce(force1)   
    print("system added with bias.")
    return system


def update_bias_2D_FC(simulation, gaussian_param, name = "BIAS", num_gaussians = 20, epsilon = 0.6):
    assert len(gaussian_param) == 43 * num_gaussians

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians:(i + 2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T

    for i in range(num_gaussians):
        current_mean = means[i][:2]
        current_covs = covs[i].reshape(6,6)
        current_covs = current_covs[:2,:2]
        # inverted_cov = np.linalg.inv(current_covs)

        simulation.context.setParameter(f"A1g{i}", A[i] * 4.184)#
    
        simulation.context.setParameter(f"x0g{i}", means[i][0]) #convert to nm
        simulation.context.setParameter(f"y0g{i}", means[i][1])

        cov_row_label = ["x0","y0"]
        cov_col_label = ["a","b"]
        for j in range(2):
            for k in range(2):
                simulation.context.setParameter(f"var_{cov_row_label[j]}_{cov_col_label[k]}{i}", current_covs[j,k])

    print("system bias updated")
    return simulation


def apply_6D_bias_FC(system, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20, epsilon = 0.1):
    assert len(gaussian_param) == 43 * num_gaussians

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians:(i+2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force1 = openmm.CustomExternalForce(energy)
    force2 = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        # inverted_cov = np.linalg.inv(covs[i].reshape(6,6) + epsilon * np.eye(6,6))   
        
        current_covs = covs[i].reshape(6,6)

        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            # energy1 = f"A1g{i}*exp(-(x-x0g{i})^2/(2*(var_x0_a{i} + var_x0_b{i} + var_x0_c{i} + var_x0_d{i} + var_x0_e{i} + var_x0_f{i})) - (y-y0g{i})^2/(2*(var_y0_a{i} + var_y0_b{i} + var_y0_c{i} + var_y0_d{i} + var_y0_e{i} + var_y0_f{i})) - (z-z0g{i})^2/(2*(var_z0_a{i} + var_z0_b{i} + var_z0_c{i} + var_z0_d{i} + var_z0_e{i} + var_z0_f{i})))" #in openmm unit, kj/mol, nm.
            energy1 = f"A1g{i}*exp(-(x-x0g{i})^2/(2*var_x0{i}) - (y-y0g{i})^2/(2*var_y0{i}) - (z-z0g{i})^2/(2*var_z0{i}))" #in openmm unit, kj/mol, nm.
            force1 = openmm.CustomExternalForce(energy1)

            # energy2 = f"A2g{i}*exp(-(x-x1g{i})^2/(2*(var_x1_a{i} + var_x1_b{i} + var_x1_c{i} + var_x1_d{i} + var_x1_e{i} + var_x1_f{i})) - (y-y1g{i})^2/(2*(var_y1_a{i} + var_y1_b{i} + var_y1_c{i} + var_y1_d{i} + var_y1_e{i} + var_y1_f{i})) - (z-z1g{i})^2/(2*(var_z1_a{i} + var_z1_b{i} + var_z1_c{i} + var_z1_d{i} + var_z1_e{i} + var_z1_f{i})))" #in openmm unit, kj/mol, nm.
            energy2 = f"A2g{i}*exp(-(x-x1g{i})^2/(2*var_x1{i}) - (y-y1g{i})^2/(2*var_y1{i}) - (z-z1g{i})^2/(2*var_z1{i}))" #in openmm unit, kj/mol, nm.
            force2 = openmm.CustomExternalForce(energy2)

        #examine the current energy term within force.

        print(force1.getEnergyFunction())
        print(force2.getEnergyFunction())

        force1.addGlobalParameter(f"A1g{i}", A[i] * 4.184) #convert to kJ/mol
        force1.addGlobalParameter(f"x0g{i}", means[i][0]) #convert to nm
        force1.addGlobalParameter(f"y0g{i}", means[i][1])
        force1.addGlobalParameter(f"z0g{i}", means[i][2])

        force1.addGlobalParameter(f"var_x0{i}", current_covs[0,0])
        force1.addGlobalParameter(f"var_y0{i}", current_covs[1,1])
        force1.addGlobalParameter(f"var_z0{i}", current_covs[2,2])

        force2.addGlobalParameter(f"A2g{i}", A[i] * 4.184) #convert to kJ/mol
        force2.addGlobalParameter(f"x1g{i}", means[i][3])
        force2.addGlobalParameter(f"y1g{i}", means[i][4])
        force2.addGlobalParameter(f"z1g{i}", means[i][5])

        force2.addGlobalParameter(f"var_x1{i}", current_covs[3,3])
        force2.addGlobalParameter(f"var_y1{i}", current_covs[4,4])
        force2.addGlobalParameter(f"var_z1{i}", current_covs[5,5])

        force1.addParticle(0)
        force2.addParticle(1)

        #we append the force to the system.
        system.addForce(force1)
        system.addForce(force2)   
    print("system added with bias.")
    return system


def update_bias_6D_FC(simulation, gaussian_param, name = "BIAS", num_gaussians = 20):
    assert len(gaussian_param) == 43 * num_gaussians

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians:(i + 2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T

    for i in range(num_gaussians):
        # inverted_cov = np.linalg.inv(covs[i].reshape(6,6) + epsilon * np.eye(6,6))

        current_covs = covs[i].reshape(6,6)

        simulation.context.setParameter(f"A1g{i}", A[i] * 4.184)#
        simulation.context.setParameter(f"A2g{i}", A[i] * 4.184)#
        simulation.context.setParameter(f"x0g{i}", means[i][0]) #convert to nm
        simulation.context.setParameter(f"y0g{i}", means[i][1])
        simulation.context.setParameter(f"z0g{i}", means[i][2])
        simulation.context.setParameter(f"x1g{i}", means[i][3]) 
        simulation.context.setParameter(f"y1g{i}", means[i][4])
        simulation.context.setParameter(f"z1g{i}", means[i][5])
        simulation.context.setParameter(f"var_x0{i}", current_covs[0,0])
        simulation.context.setParameter(f"var_y0{i}", current_covs[1,1])
        simulation.context.setParameter(f"var_z0{i}", current_covs[2,2])
        simulation.context.setParameter(f"var_x1{i}", current_covs[3,3])
        simulation.context.setParameter(f"var_y1{i}", current_covs[4,4])
        simulation.context.setParameter(f"var_z1{i}", current_covs[5,5])


    print("system bias updated")
    return simulation
