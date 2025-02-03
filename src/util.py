import time
import os
from typing import Tuple

import numpy as np
import numpy.typing as nptypes

import openmm
from openmm import unit
from openmm.unit.quantity import Quantity
from openmm.app.topology import Topology
from openmm.app.element import Element

from scipy.optimize import minimize

from MSM import *

def create_data_folders() -> dict:
    """
    This method creates folders to store reuslt data for each step in analysis.
    Folders include ``params``, ``pca_coords`` and ``plots``.
    
    Returns
    ---
    dict
        Dictonary contains the paths to created folders

    Raises
    ---
    OSError
        When failes to create any folder
    """
    print("Creating folder structure...")
    # Generate a time tag for this run
    time_tag = time.strftime("%Y%m%d-%H%M%S")
    print(f"This execuation is using time tag: {time_tag}\nIn progress data files can be found in folder ./data/{time_tag}")
    oserror_msg = "[Error] Cannot create root directory for this run, please check your permission and/or avaliable disk space."

    # Create root folder of this run
    directory_root = "./data/" + time_tag + "/"
    try:
        os.makedirs(directory_root)
    except:
        # Just in case
        raise OSError(oserror_msg)

    # Create other folders for this run
    directory_paths = [directory_root + "params", directory_root + "pca_coords", directory_root + "plots", directory_root + "trajectories"]

    try:
        for folder_name in directory_paths:
            os.makedirs(folder_name)
    except:
        # Just in case
        raise OSError(oserror_msg)
    
    result = {
        "root": directory_root,
        "params": directory_paths[0],
        "pca_coords": directory_paths[1],
        "plots": directory_paths[2],
        "trajectories": directory_paths[3]
    }
    print("Folder structure creation complete!")
    return result

def create_topology() -> Tuple[Element, Topology, Quantity]:
    """
    This method creates topology for the analysis, also creates element and mass.
    
    Returns
    ---
    Element
        The element in this analysis

    Topology
        The topology with chains and residues

    Quantity
        The mass this analysis is considering
    """
    element = Element(0, "X", "X", 1.0)

    topology = Topology()
    topology.getUnitCellDimensions == None
    topology.addChain()
    
    topology.addResidue("xxx", topology._chains[0])
    topology.addAtom("X1", element, topology._chains[0]._residues[0])
    topology.addAtom("X2", element, topology._chains[0]._residues[0])

    mass = 12.0 * unit.amu

    return element, topology, mass

def random_initial_bias_Nd(initial_position: openmm.unit.Quantity, num_gaussians = 20) -> nptypes.NDArray:
    """
    Generates a randomized initial bias using multiple Gaussian distributions.

    The function samples `num_gaussians` Gaussian distributions centered around
    `initial_position`, with means perturbed within ±1 and standard deviations
    drawn from [1.0, 5.0]. The output is a flattened array containing amplitudes,
    mean vectors, and variances.

    Parameters
    ----------
    initial_position : openmm.unit.Quantity
        The initial position in OpenMM's `md_unit_system`, used to define Gaussian centers.
    num_gaussians : int, optional
        Number of Gaussian distributions. Default is 20.
    amplitude : float, optional
        Scaling factor for the Gaussians. Default is 0.01.
    num_dimensions : int, optional
        Number of spatial dimensions. Default is 6.

    Returns
    -------
    numpy.ndarray
        A flattened array of Gaussian parameters: amplitudes, means, and variances.

    """

    num_dimensions = 6

    # Convert initial_position to OpenMM's md_unit_system and extract the first two components
    initial_position1 = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[0]
    initial_position2 = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[1]

    rng = np.random.default_rng()
    a = 0.01 * np.ones((num_gaussians, 1))
    mean_vecs = []
    all_stdevs = []

    # Generate mean vectors and standard deviations for the first half of dimensions
    for i in range(num_dimensions // 2):
        mean_vecs.append(rng.uniform(initial_position1[i] - 1, initial_position1[i] + 1, num_gaussians))
        all_stdevs.append(rng.uniform(1.0, 5.0, num_gaussians))

    # Generate mean vectors and standard deviations for the second half of dimensions
    for i in range(num_dimensions // 2):
        mean_vecs.append(rng.uniform(initial_position2[i] - 1, initial_position2[i] + 1, num_gaussians))
        all_stdevs.append(rng.uniform(1.0, 5.0, num_gaussians))

    # Convert lists to numpy arrays for efficient operations
    mean_vecs = np.array(mean_vecs).T
    all_stdevs = np.array(all_stdevs).T

    all_cov = []
    
    # Construct diagonal covariance matrices for each Gaussian
    for i in range(len(all_stdevs)):
        all_cov.append(np.diag(all_stdevs[i] ** 2).flatten())

    all_cov = np.array(all_cov)

    # Concatenate amplitudes, means, and covariance values into a flattened array
    return np.concatenate([a,mean_vecs, all_cov], axis = 1).T.flatten()

def apply_2D_bias_cov_FC(system: openmm.System, gaussian_param: nptypes.NDArray, num_gaussians = 20):
    """
    Applies a 2D Gaussian bias potential to an OpenMM system.

    This function unpacks Gaussian parameters and constructs a set of `num_gaussians` 
    2D biasing potentials with covariance matrices. Each Gaussian is added as a 
    `CustomExternalForce` to the system.

    Parameters
    ----------
    system : openmm.System
        The OpenMM system to which the bias potential is applied.
    gaussian_param : numpy.ndarray
        A 1D array containing the Gaussian parameters, expected to be of size `43 * num_gaussians`.
    num_gaussians : int, optional
        The number of Gaussian bias potentials to apply. Default is 20.

    Returns
    -------
    openmm.System
        The system with the added Gaussian bias forces.

    Notes
    -----
    - Each Gaussian is defined by an amplitude, mean positions (x, y), and a 2x2 covariance matrix.
    - The bias potential follows a bivariate Gaussian function.
    - All forces are added in OpenMM units (kJ/mol, nm).

    """
    assert len(gaussian_param) == 43 * num_gaussians

    # Unpack Gaussian parameters: Amplitude, mean positions, and covariance matrix
    # gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []

    # Extract mean values and covariance matrices from the parameter array
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians:(i+2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T
    
    # Initialize an external force for the bias potential
    energy = "0"
    force1 = openmm.CustomExternalForce(energy)

    # Loop over each Gaussian to construct the bias potential
    for i in range(num_gaussians):
        current_mean = means[i][:2]
        current_covs = covs[i].reshape(6,6)
        current_covs = current_covs[:2,:2]
        
        # Define the energy function for the 2D Gaussian potential
        energy1 = f"A1g{i}*exp(-(x-x0g{i})^2 * var_y0_b{i}/(2*(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2)) - (y-y0g{i})^2 * var_x0_a{i}/(2*(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2)) + var_x0_b{i}*(x-x0g{i})*(y-y0g{i})/(var_x0_a{i}*var_y0_b{i}-var_x0_b{i}^2))"        
        force1 = openmm.CustomExternalForce(energy1)

        # Add global parameters to the OpenMM force object
        force1.addGlobalParameter(f"A1g{i}", A[i] * 4.184)
        force1.addGlobalParameter(f"x0g{i}", current_mean[0])
        force1.addGlobalParameter(f"y0g{i}", current_mean[1])
        
        # Add covariance matrix elements as global parameters
        cov_row_label = ["x0","y0"]
        cov_col_label = ["a","b"]
        for j in range(2):
            for k in range(2):
                force1.addGlobalParameter(f"var_{cov_row_label[j]}_{cov_col_label[k]}{i}", current_covs[j,k])
        
        force1.addParticle(0)

        # Append the force to the OpenMM system
        system.addForce(force1)   
    print("Bias added to system")
    return system

def get_working_MM(M: MSM):
    """
    Gives the working matrix and the
    corresponding indices of the original
    markov matrix
    """
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def gaussian_2D(x, y, a, bx, by, cx, cy):
    return a * np.exp(-(((x - bx) ** 2 / 2 / cx ** 2) + ((y - by) ** 2 / 2 / cy ** 2)))

def gaussian_2d_cov(x, y, gaussian_param):
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

def get_total_bias_2d(x, y, gaussian_params, no_covariance = True):
    """
    here we get the total bias at x,y.
    note: we used the transposed K matrix, we need to apply transposed total gaussian bias.
    """
    N = x.shape[0] #N is the number of grid points.
    total_bias = np.zeros((N, N))
    if no_covariance:
        num_gaussians = len(gaussian_params) // 5
    
        a = gaussian_params[:num_gaussians]
        bx = gaussian_params[num_gaussians:2 * num_gaussians]
        by = gaussian_params[2 * num_gaussians:3 * num_gaussians]
        cx = gaussian_params[3 * num_gaussians:4 * num_gaussians]
        cy = gaussian_params[4 * num_gaussians:5 * num_gaussians]

        for i in range(num_gaussians):
            total_bias = total_bias + gaussian_2D(x, y, a[i], bx[i], by[i], cx[i], cy[i])
    else:
        num_gaussians = gaussian_params.shape[0]
        for i in range(num_gaussians):
            total_bias += gaussian_2d_cov(x, y, gaussian_params[i])
    return total_bias

def try_and_optim_M(M, X, working_indices, prop_index, N = 20, num_gaussian = 20, start_index = 0, end_index = 0, paths: dict = None):
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
    start_state_working_index_xy = np.unravel_index(working_indices[start_state_working_index], (N, N), order = 'F')
    end_state_working_index_xy = np.unravel_index(working_indices[end_state_working_index], (N, N), order = 'C')
    print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)

    #now our M/working_indices could be incontinues. #N = M.shape[0]
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    #here we find the x,y maximum and minimun in xy coordinate space, with those working index
    #we use this to generate the random gaussian params.
    working_indices_xy = np.unravel_index(working_indices, (N, N), order = 'C')

    msm = MSM()
    msm.qspace = np.stack((X[0], X[1]), axis = 1)
    msm.build_MSM_from_M(M, dim = 1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
    msm._compute_peq_fes_M()

    #parameter settings.
    pca_x_min = X[0][0]
    pca_x_max = X[0][-1]
    pca_y_min = X[1][0]
    pca_y_max = X[1][-1]

    c_low_factor = 0.1 / (2 * np.pi) #defines the factor used in definition of lower barrier of cx.
    c_high_factor = 1 / (2 * np.pi) 


    X_mesh, Y_mesh = np.meshgrid(X[0], X[1])
    
    for try_num in range(100):
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

        if try_num % 100 == 0:
            msm._kemeny_constant_check()
            print("random try:", try_num, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = gaussian_params

    print("best mfpt:", best_mfpt)
    
    best_bias = get_total_bias_2d(X_mesh, Y_mesh, best_params)
    #here we plot the total bias just to check the bias applied in pca space make sence.
    plt.figure()
    plt.imshow(best_bias,cmap="coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin="lower", aspect='auto')
    plt.title("best_bias in the pca space.")
    plt.colorbar()
    plt.savefig(f'{paths["plots"]}/test_best_bias_p{prop_index}.png')
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

    res = minimize(
        mfpt_helper,
        best_params,
        args = (
            M,
            start_state_working_index,
            end_state_working_index,
            working_indices_xy
            ),
        method = "L-BFGS-B",
        bounds = [(0.1, 1.1)] * num_gaussian + [(pca_x_min, pca_x_max)] * num_gaussian + [(pca_y_min, pca_y_max)] * num_gaussian + [(c_low_factor * (pca_x_max - pca_x_min),c_high_factor * (pca_x_max - pca_x_min))] * num_gaussian + [(c_low_factor * (pca_y_max - pca_y_min), c_high_factor * (pca_y_max - pca_y_min))] * num_gaussian,
        tol=1e-4
        )
    
    gaussian_param = res.x

    optimised_bias = get_total_bias_2d(X_mesh, Y_mesh, gaussian_param)
    plt.figure()
    plt.imshow(optimised_bias, cmap = "coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin = "lower", aspect='auto')
    plt.title("optimised_bias in the pca space.")
    plt.colorbar()
    plt.savefig(f'{paths["plots"]}/test_optimised_bias_p{prop_index}.png')
    plt.close()

    A = gaussian_param[:num_gaussian]
    x0 = gaussian_param[num_gaussian:2 * num_gaussian]
    y0 = gaussian_param[2 * num_gaussian:3 * num_gaussian]
    sigma_x = gaussian_param[3 * num_gaussian:4 * num_gaussian]
    sigma_y = gaussian_param[4 * num_gaussian:5 * num_gaussian]

    mean_vec = []
    cov_matrix = []
    for i in range(num_gaussian):
        mean_vec.append(np.array([x0[i], y0[i]]))
        cov_matrix.append(np.array([[sigma_x[i] ** 2, 0], [0, sigma_y[i] ** 2]]))

    # Also temporarily return best_params
    return res.x, np.array(mean_vec), np.array(cov_matrix)

def find_closest_index(working_indices, final_index, N):
    """
    returns the farest index in 1D.

    here we find the closest state to the final state.
    first we unravel all the index to 2D.
    then we use the lowest RMSD distance to find the closest state.
    then we ravel it back to 1D.
    note: for now we only find the first-encounted closest state.
          we can create a list of all the closest states, and then choose random one.
    """
    def rmsd_dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    working_x, working_y = np.unravel_index(working_indices, (N, N), order = 'C')
    working_states = np.stack((working_x, working_y), axis = 1)
    final_state = np.unravel_index(final_index, (N,N), order = 'C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N, N), order = 'C')
    return closest_index

def update_bias_2D_FC(simulation, gaussian_param, num_gaussians = 20):
    assert len(gaussian_param) == 43 * num_gaussians

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    A = gaussian_param[:num_gaussians]
    means = []
    covs = []
    for i in range(6):
        means.append(gaussian_param[(i + 1) * num_gaussians : (i + 2) * num_gaussians])
        for j in range(6):
            covs.append(gaussian_param[(7 + 6 * i) * num_gaussians + j * num_gaussians : (7 + 6 * i) * num_gaussians + (j + 1) * num_gaussians])
    means = np.array(means).T
    covs = np.array(covs).T

    for i in range(num_gaussians):
        current_covs = covs[i].reshape(6,6)
        current_covs = current_covs[:2,:2]

        simulation.context.setParameter(f"A1g{i}", A[i] * 4.184)#
    
        simulation.context.setParameter(f"x0g{i}", means[i][0]) #convert to nm
        simulation.context.setParameter(f"y0g{i}", means[i][1])

        cov_row_label = ["x0", "y0"]
        cov_col_label = ["a", "b"]
        for j in range(2):
            for k in range(2):
                simulation.context.setParameter(f"var_{cov_row_label[j]}_{cov_col_label[k]}{i}", current_covs[j,k])

    print("system bias updated")
    return simulation