import config
import fes_functions

import numpy.typing as nptypes

import matplotlib.pyplot as plt

import tqdm

import openmm
import openmm.app
from openmm.app import Simulation
from openmm import unit

import numpy as np

from sklearn.decomposition import PCA

import pickle as pk

def pca_transform(data: np.ndarray, pca_fitting_file_path:str = None, mode = "to_pca"):
    """
    Applies PCA transformation or its inverse using a pre-fitted PCA model.

    Loads a pre-trained PCA model from the given file and applies different 
    transformations based on the specified mode.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be transformed.
    pca_fitting_file_path : str
        Path to the pre-fitted PCA model file (must be provided).
    mode : str, optional
        The transformation mode (default: "to_pca"). Available modes:
        - "to_pca" : Apply PCA transformation.
        - "from_pca" : Apply inverse PCA transformation.
        - "matrix_to_pca" : Project a covariance matrix to PCA space.
        - "matrix_from_pca" : Project a covariance matrix from PCA space.

    Returns
    -------
    numpy.ndarray
        The transformed data.

    Raises
    ------
    ValueError
        If `pca_fitting_file_path` is not provided or `mode` is invalid.

    Notes
    -----
    - This function assumes that the PCA model was trained using `sklearn.decomposition.PCA`.
    - The `matrix_to_pca` and `matrix_from_pca` modes use PCA components for 
      matrix projection.
    """

    if pca_fitting_file_path == None:
        raise ValueError("PCA Fitting File cannot be empty!")
    
    # Load the pre-trained PCA model
    pca_reload = pk.load(open(pca_fitting_file_path, "rb"))

    # Transform data to PCA space
    if mode == "to_pca":    
        return pca_reload.transform(data)
    
    # Inverse transform data back to original space
    elif mode == "from_pca":
        return pca_reload.inverse_transform(data)
    
    # Project a covariance matrix into PCA space
    elif mode == "matrix_to_pca":
        return np.dot(pca_reload.components_,np.dot(data, pca_reload.components_.T))
    
    # Project a covariance matrix back from PCA space
    elif mode == "matrix_from_pca":
        return np.dot(pca_reload.components_.T,np.dot(data, pca_reload.components_))
    
    # Handle invalid mode input
    else:
        raise ValueError(f"Invalid Mode Detected: {mode}!")

def prop_pca(
        simulation: Simulation, 
        prop_index: int, 
        pos_traj: nptypes.NDArray, 
        pos_traj_pca: nptypes.NDArray, 
        coor_xy_list: list = None,
        steps = config.propagation_step,
        dcdfreq = config.dcdfreq_mfpt,
        num_bins = config.num_bins,
        paths: dict = None,
        i_prop: int = 0,
        pca = {'mode':'to_pca'}):
    """
    Use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    Use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """ 

    # Setup additional config for pca
    pca['max_dim'] = 6
    pca['target_dim'] = 2

    file_handle = open(f"{paths["trajectories"]}/langevin_sim_explore_{prop_index}.dcd", 'bw')

    all_coor = np.zeros(int(steps/dcdfreq)).astype(object)
    for i in tqdm.tqdm(range(int(steps / dcdfreq)), desc = f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        # Deleted paramater enforcePeriodicBox for getState
        state = simulation.context.getState(getPositions = True)
        state_pos = state.getPositions(asNumpy = True)
        try:
            all_coor[i] = state_pos.value_in_unit(unit.nanometers)
        except:
            print(i)
            print(int(config.propagation_step / config.dcdfreq))
            raise KeyError("np.zeros array too small to contain full propagation!!")
    file_handle.close()

    min_distance = np.inf
    closest_value = None
    for i in range(len(all_coor)):
        all_coor[i] = all_coor[i].reshape(pca['max_dim']) ### formatting data before PCA fit. Use opportunity to calculate distance to end state
        distance = np.linalg.norm(all_coor[i] - config.end_state_array)
        if distance < min_distance:
            min_distance = distance
            closest_value = all_coor[i]
    all_coor = np.vstack(all_coor)

    coor_xy_original = all_coor.squeeze()[:,:2]
    x = np.linspace(0,2 * np.pi, config.num_bins)
    y = x.copy()
    coor_x_digitized_original = np.digitize(coor_xy_original[:, 0], x) - 1#quick fix for digitized to 0 or maximum error. #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    coor_y_digitized_original = np.digitize(coor_xy_original[:, 1], y) - 1

    coor_xy_digitized_original = np.stack([coor_x_digitized_original, coor_y_digitized_original], axis=1) #shape: [all_frames, 2]
    coor_xy_digitized_ravel_original = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order = 'F') for coor_temp in coor_xy_digitized_original])
    pos_traj[prop_index,:] =  coor_xy_digitized_ravel_original.astype(np.int64)

    # Digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    # Save the top to pdb.
    with open(f"{paths["trajectories"]}/langevin_sim_explore_{prop_index}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    # Find the max and min for the past recorded coor
    if prop_index > 0:
        max0 = max([np.max(element[:, 0]) for element in coor_xy_list])
        max1 = max([np.max(element[:, 1]) for element in coor_xy_list])
        min0 = min([np.min(element[:, 0]) for element in coor_xy_list])
        min1 = min([np.min(element[:, 1]) for element in coor_xy_list])

    if pca['mode'] == "pca_transform":
        num = pca["target_dim"]
        pca_ = PCA(n_components = 2)
        principal_variance = pca_.fit(all_coor)
        pk.dump(principal_variance, open(f"{paths["pca_coords"]}/{prop_index}.pkl", "wb"))
        coor_transform = pca_transform(all_coor, f"{paths["pca_coords"]}/{prop_index}.pkl", mode="to_pca")
        
        
        coor_xy = coor_transform.squeeze()[:,:pca['target_dim']] 
        epsilon = 1e-10

        # Compare the past and current max and min
        if prop_index == 0:
            pca_start = [min(coor_xy[:, 0]) - epsilon,min(coor_xy[:, 1] - epsilon)] #arbitary number for pca space discretization.
            pca_end = [max(coor_xy[:, 0] + epsilon), max(coor_xy[:, 1] + epsilon)]
        else:
            max0 = max(max0, max(coor_xy[:, 0]))
            max1 = max(max1, max(coor_xy[:, 1])) 
            min0 = min(min0, min(coor_xy[:, 0]))       
            min1 = min(min1, min(coor_xy[:, 1]))       

            pca_start = [min0 - epsilon, min1 - epsilon]
            pca_end = [max0 + epsilon, max1 + epsilon]


        # Save the digitised coor and boundary X
        coor_digitised = []
        X = []
        for  i in range(pca["target_dim"]):
            x = np.linspace(pca_start[i], pca_end[i], num_bins)
            # find midpoints of bins
            X.append(x)
            coor_digitised.append(np.digitize(coor_xy[:, i], x) - 1)
        
        
        coor_X_digitised = np.stack(coor_digitised, axis = 1)
        closest_value_pca = pca_transform(closest_value.reshape(1, 6), f"{paths['pca_coords']}/0.pkl").reshape(pca["target_dim"])
        target_value_pca = pca_transform(config.end_state_array.reshape(1, 6), f"{paths['pca_coords']}/0.pkl").reshape(pca["target_dim"])
        coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order = 'F') for coor_temp in coor_X_digitised])


    elif pca['mode'] == "to_pca":
        coor_transform = pca_transform(all_coor, f"{paths['pca_coords']}/0.pkl")
        coor_xy = coor_transform.squeeze()[:,:pca['target_dim']]

        # pc1, pc2 should be discretized in their own min/max +- a epsilon.
        epsi = 1e-10

        # Compare the past and current max and min
        if prop_index == 0:
            pca_start = [min(coor_xy[:, 0]) - epsi, min(coor_xy[:, 1]) - epsi]
            pca_end = [max(coor_xy[:, 0]) + epsi, max(coor_xy[:, 1]) + epsi]
        else:
            max0 = max(max0, max(coor_xy[:, 0]))
            max1 = max(max1, max(coor_xy[:, 1])) 
            min0 = min(min0, min(coor_xy[:, 0]))       
            min1 = min(min1, min(coor_xy[:, 1]))       

            pca_start = [min0 - epsi, min1 - epsi]
            pca_end = [max0 + epsi, max1 + epsi]


        # Save the digitised coor and boundary X
        coor_digitised = []
        X = []
        for  i in range(pca["target_dim"]):
            x = np.linspace(pca_start[i], pca_end[i], num_bins)
            X.append(x)
            coor_digitised.append(np.digitize(coor_xy[:, i], x) - 1)
        coor_X_digitised = np.stack(coor_digitised,axis = 1)
        closest_value_pca = pca_transform(closest_value.reshape(1, 6),f"{paths['pca_coords']}/0.pkl").reshape(pca["target_dim"])

        # add target value
        target_value_pca = pca_transform(config.end_state_array.reshape(1, 6), f"{paths['pca_coords']}/0.pkl").reshape(pca["target_dim"])

        coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order = 'F') for coor_temp in coor_X_digitised])
    else:
        raise ValueError("Please Input a valid PCA configuration!")
    
    # we append the coor_xy_digitized into the pos_traj.
    print("The shape of coor_xy_digitised_ravel is:", coor_xy_digitized_ravel.shape)
    pos_traj_pca[prop_index,:] = coor_xy_digitized_ravel.astype(np.int64)

    coor_xy_digital_ravelled_total = pos_traj_pca[:prop_index+1,:] #shape: [prop_index+1, all_frames * 1]
    coor_xy_digital_ravelled_total = coor_xy_digital_ravelled_total.reshape(-1, 1) #shape: [prop_index+1 * all_frames, 1]

    if prop_index == 0:
        orig_gaussian_params = np.loadtxt(f"{paths["params"]}/gaussian_fes_param_0.txt").reshape(-1, config.num_gaussian).T
        pca_gaussian_params = np.zeros((config.num_gaussian,pca["target_dim"] + pca["target_dim"] ** 2 + 1))
        for i in range(config.num_gaussian):
            A = orig_gaussian_params[i][0]
            mean_vec = orig_gaussian_params[i][1:pca["max_dim"] + 1].copy()
            cov_mat = orig_gaussian_params[i][pca["max_dim"] + 1:].reshape(pca['max_dim'], pca['max_dim']).copy()
            mean_vec_pca = pca_transform(mean_vec.reshape(1, -1), f"{paths['pca_coords']}/{prop_index}.pkl", mode = "to_pca")
            cov_mat_pca = pca_transform(cov_mat, f"{paths['pca_coords']}/{prop_index}.pkl", mode = "matrix_to_pca")
            pca_gaussian_params[i] = np.concatenate([np.array([A]), mean_vec_pca.reshape(pca["target_dim"]), cov_mat_pca.flatten()])
        np.savetxt(f"{paths["params"]}/gaussian_fes_param_pca_0.txt", pca_gaussian_params.T.flatten())
    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 7])

    for i in range(prop_index+1):
        temp_params = np.loadtxt(f"{paths["params"]}/gaussian_fes_param_pca_{i}.txt").reshape(-1,config.num_gaussian).T

        gaussian_params[i,:,:] = temp_params
        print(f"gaussian_params for propagation {i} loaded.")

    print("DHAM input is: ", coor_xy_digital_ravelled_total.reshape(prop_index + 1, -1, 1).shape)

    F_M, MM = fes_functions.DHAM_it(coor_xy_digital_ravelled_total.reshape(prop_index+1, -1, 1), gaussian_params = gaussian_params, X = X, T = 300, lagtime = 1, numbins = num_bins, paths = paths, prop_index = prop_index)
    ur_pos = coor_xy_digital_ravelled_total[-1] #the current position of the particle, in ravelled 1D form.

    # determine if the particle has reached the target state.

    # we want to measure distance in terms of original space
    
    end_state = config.end_state_array
    reach = None
    for index_d, d in enumerate(all_coor):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state)
        if target_distance < 0.1:
            reach = index_d * config.dcdfreq_mfpt

    #on-the-fly checking plots.
    plt.figure()
    plt.plot(coor_xy[:, 0], coor_xy[:, 1])
    plt.plot(closest_value_pca[0], closest_value_pca[1], marker = 'o', color = 'red', label = 'closest point')
    plt.plot(coor_xy[0, 0], coor_xy[0, 1], marker = 'o', color = 'blue', label = 'start point')
    plt.title("traj and closest in pca space.")
    plt.legend()
    plt.savefig(f"{paths["plots"]}/test_pca_p{i_prop}.png")
    plt.close()

    # plot the sampled FES on the PC1/PC2 space.
    plt.figure()
    plt.imshow(np.reshape(F_M.astype(float), [config.num_bins, config.num_bins], order = 'C'), extent = [min(X[0]), max(X[0]), min(X[1]), max(X[1])], cmap = 'coolwarm', origin = 'lower', aspect='auto') 
    plt.colorbar()
    plt.savefig(f"{paths["plots"]}/test_fes_p{i_prop}.png")
    plt.close()

    return pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy