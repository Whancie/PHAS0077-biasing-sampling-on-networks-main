#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm.app.topology import Topology
from openmm.app.element import Element
import mdtraj
import csv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from dham import *
from MSM import MSM
import pickle as pk
from util import random_initial_bias_2d, apply_bias, try_and_optim_M, update_bias,apply_fes, pca_transform, random_initial_bias_Nd, get_total_bias_2d, get_total_bias_6D, apply_2D_bias_cov_FC, update_bias_2D_FC, apply_6D_bias_FC, update_bias_6D_FC


def prop_pca(simulation,
              prop_index, 
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              pos_traj_pca,
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq_mfpt,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              time_tag = None,
              reach=None,
              pca = {'mode':'to_pca'}):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """ 
    try:
        pca['max_dim']
    except:
        pca['max_dim'] = 6
    try:
        pca['target_dim']
    except:
        pca['target_dim'] = 2
    
    file_handle = open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", 'bw')
    #dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is no longer a global pararm, we need pass this.
    all_coor = np.zeros(int(steps/dcdfreq)).astype(object)
    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        state_pos = state.getPositions(asNumpy=True)
        #dcd_file.writeModel(state_pos)
        try:
            all_coor[_] = state_pos.value_in_unit(unit.nanometers)
        except:
            print(_)
            print(int(config.propagation_step/config.dcdfreq))
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
    x = np.linspace(0,2*np.pi,config.num_bins)
    y = x.copy()
    coor_x_digitized_original = np.digitize(coor_xy_original[:,0], x) - 1#quick fix for digitized to 0 or maximum error. #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    coor_y_digitized_original = np.digitize(coor_xy_original[:,1], y) - 1

    coor_xy_digitized_original = np.stack([coor_x_digitized_original, coor_y_digitized_original], axis=1) #shape: [all_frames, 2]
    coor_xy_digitized_ravel_original = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order='F') for coor_temp in coor_xy_digitized_original])
    pos_traj[prop_index,:] =  coor_xy_digitized_ravel_original.astype(np.int64)


    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    #save the top to pdb.
    with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


    # Find the max and min for the past recorded coor
    if prop_index > 0:
        max0 = max([np.max(element[:,0]) for element in coor_xy_list])
        max1 = max([np.max(element[:,1]) for element in coor_xy_list])
        min0 = min([np.min(element[:,0]) for element in coor_xy_list])
        min1 = min([np.min(element[:,1]) for element in coor_xy_list])



    if pca['mode'] == "pca_transform":
        num = pca["target_dim"]
        pca_ = PCA(n_components=2)
        principal_variance = pca_.fit(all_coor)
        pk.dump(principal_variance, open(f"./pca_coords/{time_tag}_pca_{prop_index}.pkl", "wb"))
        coor_transform = pca_transform(all_coor, f"./pca_coords/{time_tag}_pca_{prop_index}.pkl", mode="to_pca")
        # pca_end = pca_transform(convert_6D_pos_array(config.end_state.value_in_unit_system(openmm.unit.md_unit_system)),f"./pca_coords/{time_tag}_pca_{prop_index}.pkl", mode="to_pca").reshape(pca["target_dim"])
        # pca_start = pca_transform(convert_6D_pos_array(config.start_state.value_in_unit_system(openmm.unit.md_unit_system)),f"./pca_coords/{time_tag}_pca_{prop_index}.pkl", mode="to_pca").reshape(pca["target_dim"])
        # pca_dim1_range = [-2 + min(pca_start[0],pca_end[0]), max(pca_start[0],pca_end[0]) + 2]
        # pca_dim2_range = [-2 + min(pca_start[1],pca_end[1]), max(pca_start[1],pca_end[1]) + 2]
        #pca_dim_range = [pca_dim1_range,pca_dim2_range]
        coor_xy = coor_transform.squeeze()[:,:pca['target_dim']] 
        epsilon = 1e-10

        # Compare the past and current max and min
        if prop_index == 0:
            pca_start = [min(coor_xy[:,0]) - epsilon,min(coor_xy[:,1] - epsilon)] #arbitary number for pca space discretization.
            pca_end = [max(coor_xy[:,0] + epsilon),max(coor_xy[:,1] + epsilon)]
        else:
            max0 = max(max0, max(coor_xy[:,0]))
            max1 = max(max1, max(coor_xy[:,1])) 
            min0 = min(min0, min(coor_xy[:,0]))       
            min1 = min(min1, min(coor_xy[:,1]))       

            pca_start = [min0-epsilon, min1-epsilon]
            pca_end = [max0+epsilon, max1+epsilon]


        # Save the digitised coor and boundary X
        coor_digitised = []
        X = []
        for  i in range(pca["target_dim"]):
            x = np.linspace(pca_start[i],pca_end[i],num_bins)
            # find midpoints of bins
            X.append(x)

            coor_digitised.append(np.digitize(coor_xy[:,i],x) - 1)
        


        coor_X_digitised = np.stack(coor_digitised,axis=1)
        closest_value_pca = pca_transform(closest_value.reshape(1,6),f"./pca_coords/{time_tag}_pca_0.pkl").reshape(pca["target_dim"])
        target_value_pca = pca_transform(config.end_state_array.reshape(1,6),f"./pca_coords/{time_tag}_pca_0.pkl").reshape(pca["target_dim"])
        coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order='F') for coor_temp in coor_X_digitised])
    
    elif pca['mode'] == "to_pca":
        # pca_end = pca_transform(convert_6D_pos_array(config.end_state.value_in_unit_system(openmm.unit.md_unit_system)),f"./pca_coords/{time_tag}_pca_0.pkl", mode="to_pca").reshape(pca["target_dim"])
        # pca_start = pca_transform(convert_6D_pos_array(config.start_state.value_in_unit_system(openmm.unit.md_unit_system)),f"./pca_coords/{time_tag}_pca_0.pkl", mode="to_pca").reshape(pca["target_dim"])
        coor_transform = pca_transform(all_coor, f"./pca_coords/{time_tag}_pca_0.pkl")
        coor_xy = coor_transform.squeeze()[:,:pca['target_dim']]

        # pc1, pc2 should be discretized in their own min/max +- a epsilon.
        epsi = 1e-10

        # Compare the past and current max and min
        if prop_index == 0:
            pca_start = [min(coor_xy[:,0]) - epsi, min(coor_xy[:,1]) - epsi]
            pca_end = [max(coor_xy[:,0]) + epsi,max(coor_xy[:,1]) + epsi]
        else:
            max0 = max(max0, max(coor_xy[:,0]))
            max1 = max(max1, max(coor_xy[:,1])) 
            min0 = min(min0, min(coor_xy[:,0]))       
            min1 = min(min1, min(coor_xy[:,1]))       

            pca_start = [min0-epsi, min1-epsi]
            pca_end = [max0+epsi, max1+epsi]


        # Save the digitised coor and boundary X
        coor_digitised = []
        X = []
        for  i in range(pca["target_dim"]):
            x = np.linspace(pca_start[i],pca_end[i],num_bins)
            #x += (x[1] - x[0])/2
            #x = np.linspace(min(coor_xy[:,i]), max(coor_xy[:,i]),num_bins - 1)
            X.append(x)
            coor_digitised.append(np.digitize(coor_xy[:,i],x) - 1)
        coor_X_digitised = np.stack(coor_digitised,axis=1)
        closest_value_pca = pca_transform(closest_value.reshape(1,6),f"./pca_coords/{time_tag}_pca_0.pkl").reshape(pca["target_dim"])

        # add target value
        target_value_pca = pca_transform(config.end_state_array.reshape(1,6),f"./pca_coords/{time_tag}_pca_0.pkl").reshape(pca["target_dim"])

        coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order='F') for coor_temp in coor_X_digitised])
    else:
        raise ValueError("Please Input a valid PCA configuration!")
        
    # Plot real traj test.
    if False:
        coor_xy_digitized_ravel_unravel = np.array([np.unravel_index(x, (num_bins, num_bins), order='C') for x in coor_xy_digitized_ravel]) #shape: [all_frames, 2]

        x,y = np.meshgrid(np.linspace(0, 2*np.pi, num_bins), np.linspace(0, 2*np.pi, num_bins))

        plt.figure()
        #
        plt.xlim([0, 2*np.pi])
        plt.ylim([0, 2*np.pi])
        plt.savefig("./test.png")
        plt.close()

    # we append the coor_xy_digitized into the pos_traj.
    print("The coor_xy_digitised_ravel is:", coor_xy_digitized_ravel, "with shape", coor_xy_digitized_ravel.shape)
    pos_traj_pca[prop_index,:] = coor_xy_digitized_ravel.astype(np.int64) # Put in first two elements of original coordinates

    #pos_traj_pca = pca_transform(pos_traj[:prop_index+1,:],pca_fitting_file_path=f"./pca_coords/{time_tag}_pca_{"0" if pca['mode'] == "to_pca" else prop_index}.pkl")
    #we take all previous ravelled position from pos_traj and append it to the total list, feed into the DHAM.
    coor_xy_digital_ravelled_total = pos_traj_pca[:prop_index+1,:] #shape: [prop_index+1, all_frames * 1]
    coor_xy_digital_ravelled_total = coor_xy_digital_ravelled_total.reshape(-1,1) #shape: [prop_index+1 * all_frames, 1]

    # here we load all the gaussian_params from previous propagations.
    # size of gaussian_params: [num_propagation, num_gaussian, 3] (a,b,c),
    # note for 2D this would be [num_propagation, num_gaussian, 5] (a,bx,by,cx,cy)
    
    if prop_index == 0:
        print(f"./params/{time_tag}_gaussian_fes_param_0.txt")
        orig_gaussian_params = np.loadtxt(f"./params/{time_tag}_gaussian_fes_param_0.txt").reshape(-1,config.num_gaussian).T
        pca_gaussian_params = np.zeros((config.num_gaussian,pca["target_dim"] + pca["target_dim"]**2 + 1))
        for i in range(config.num_gaussian):
            A = orig_gaussian_params[i][0]
            mean_vec = orig_gaussian_params[i][1:pca["max_dim"] + 1].copy()
            cov_mat = orig_gaussian_params[i][pca["max_dim"] + 1:].reshape(pca['max_dim'],pca['max_dim']).copy()
            mean_vec_pca = pca_transform(mean_vec.reshape(1,-1), f"./pca_coords/{time_tag}_pca_{prop_index}.pkl",mode = "to_pca")
            cov_mat_pca = pca_transform(cov_mat, f"./pca_coords/{time_tag}_pca_{prop_index}.pkl", mode = "matrix_to_pca")
            pca_gaussian_params[i] = np.concatenate([np.array([A]),mean_vec_pca.reshape(pca["target_dim"]), cov_mat_pca.flatten()])
        np.savetxt(f"./params/{time_tag}_gaussian_fes_param_pca_0.txt", pca_gaussian_params.T.flatten())
    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 7])
    for i in range(prop_index+1):
        temp_params = np.loadtxt(f"./params/{time_tag}_gaussian_fes_param_pca_{i}.txt").reshape(-1,config.num_gaussian).T

        gaussian_params[i,:,:] = temp_params
        print(f"gaussian_params for propagation {i} loaded.")

    
    print("DHAM input is: ", coor_xy_digital_ravelled_total.reshape(prop_index+1, -1, 1).shape)

    F_M, MM = DHAM_it(coor_xy_digital_ravelled_total.reshape(prop_index+1, -1, 1), gaussian_params, X, T=300, lagtime=1, numbins=num_bins, time_tag=time_tag, prop_index=prop_index)
    cur_pos = coor_xy_digital_ravelled_total[-1] #the current position of the particle, in ravelled 1D form.
    
    # determine if the particle has reached the target state.

    # we want to measure distance in terms of original space
    
    end_state = config.end_state_array
    #end_state_xy = end_state_xyz[:2]
    for index_d, d in enumerate(all_coor):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state)
        if target_distance < 0.1:
            reach = index_d * config.dcdfreq_mfpt


    
    # current_traj = pos_traj_pca[prop_index]
    # most_visited_state = np.argmax(np.bincount(current_traj.astype(int))) #this is in digitized, ravelled form.
    # unravel_state = np.unravel_index(most_visited_state, (config.num_bins, config.num_bins), order='F')


    #on-the-fly checking plots. disable when production.
    if True: 
        #plot in pca space.
        plt.figure()
        plt.plot(coor_xy[:,0], coor_xy[:,1])
        plt.plot(closest_value_pca[0], closest_value_pca[1], marker='o', color='red', label='closest point')
        plt.plot(coor_xy[0,0], coor_xy[0,1], marker='o', color='blue', label='start point')
        # plt.plot(coor_xy[-1,0], coor_xy[-1,1], marker='x', color='orange', label='end point')
        # plt.plot(X[0][unravel_state[0]], X[1][unravel_state[1]], marker='o', color='green', label='most visited point')
        #plt.plot(target_value_pca[0], target_value_pca[1], marker='x', color='red')
        plt.title("traj and closest in pca space.")
        plt.legend()
        plt.savefig(f"./test_pca_p{i_prop}.png")
        plt.close()

        # plt.figure()
        # plt.plot(coor_xy[:,0], coor_xy[:,1])
        # plt.plot(closest_value_pca[0], closest_value_pca[1], marker='o', color='red')
        # plt.plot(target_value_pca[0], target_value_pca[1], marker='x', color='red')
        # plt.title("traj and closest/target in pca space.")
        # plt.savefig(f"./test_pca_p{i_prop}_target.png")
        # plt.close()

        # # plot in real space.
        # plt.figure()
        # plt.plot(all_coor[:,0], all_coor[:,1])
        # plt.plot(closest_value[0], closest_value[1], marker='o', color='red')
        # plt.plot(config.end_state_array[0],config.end_state_array[1], marker='o', color='red')
        # plt.title("traj and closest/target in real space.")
        # plt.savefig(f"./test_real_p{i_prop}.png")
        # plt.close()

        # plot the sampled FES on the PC1/PC2 space.
        plt.figure()
        plt.imshow(np.reshape(F_M.astype(float), [config.num_bins, config.num_bins], order = 'C'), extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], cmap = 'coolwarm', origin='lower') 
        plt.colorbar()
        plt.savefig(f"./test_fes_p{i_prop}.png")
        plt.close()


    return cur_pos, pos_traj,pos_traj_pca, MM, reach, F_M,X, closest_value_pca, coor_xy

def propagate(simulation,
              prop_index, 
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq_mfpt,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              time_tag = None,
              top=None,
              reach=None
              ):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    
    file_handle = open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", 'bw')
    dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is no longer a global pararm, we need pass this.
    all_coor = np.zeros(int(config.propagation_step/config.dcdfreq)).astype(object)
    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        state_pos = state.getPositions(asNumpy=True)
        dcd_file.writeModel(state_pos)
        all_coor[_] = state_pos.value_in_unit(unit.nanometers)
    file_handle.close()
    for i in range(len(all_coor)):
        all_coor[i] = all_coor[i].reshape(6)
    all_coor = np.vstack(all_coor)  

    #save the top to pdb.
    with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    
    #we load the pdb and pass it to mdtraj_top
    mdtraj_top = mdtraj.load(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb")

    #use mdtraj to get the coordinate of the particle.
    traj = mdtraj.load_dcd(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", top = mdtraj_top)#top = mdtraj.Topology.from_openmm(top)) #this will yield error because we using imaginary element X.
    coor = traj.xyz[:,0,:] #[all_frames,particle_index,xyz] # we grep the particle 0.

    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    x = np.linspace(0, 2*np.pi, num_bins) #hardcoded.
    y = np.linspace(0, 2*np.pi, num_bins)
    #we digitize the coor into the meshgrid.
    coor_xy = coor.squeeze()[:,:2] #we only take the x, y coordinate.
    coor_x_digitized = np.digitize(coor_xy[:,0], x)#quick fix for digitized to 0 or maximum error. #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    coor_y_digitized = np.digitize(coor_xy[:,1], y)
    coor_xy_digitized = np.stack([coor_x_digitized, coor_y_digitized], axis=1) #shape: [all_frames, 2]

    #changed order = F, temporary fix for the DHAM?
    #print(x)
    coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order='F') for coor_temp in coor_xy_digitized]) #shape: [all_frames,]

    #we test.
    if False:
        coor_xy_digitized_ravel_unravel = np.array([np.unravel_index(x, (num_bins, num_bins), order='C') for x in coor_xy_digitized_ravel]) #shape: [all_frames, 2]

        x,y = np.meshgrid(np.linspace(0, 2*np.pi, num_bins), np.linspace(0, 2*np.pi, num_bins))

        plt.figure()
        #
        plt.xlim([0, 2*np.pi])
        plt.ylim([0, 2*np.pi])
        plt.savefig("./test.png")
        plt.close()
    #we append the coor_xy_digitized into the pos_traj.
    pos_traj[prop_index,:] = coor_xy_digitized_ravel.astype(np.int64)

    #we take all previous ravelled position from pos_traj and append it to the total list, feed into the DHAM.
    coor_xy_digital_ravelled_total = pos_traj[:prop_index+1,:] #shape: [prop_index+1, all_frames * 1]
    coor_xy_digital_ravelled_total = coor_xy_digital_ravelled_total.reshape(-1,1) #shape: [prop_index+1 * all_frames, 1]

    #here we load all the gaussian_params from previous propagations.
    #size of gaussian_params: [num_propagation, num_gaussian, 3] (a,b,c),
    # note for 2D this would be [num_propagation, num_gaussian, 5] (a,bx,by,cx,cy)
    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 5])
    for i in range(prop_index+1):
        gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_fes_param_{i}.txt").reshape(-1,5)
        print(f"gaussian_params for propagation {i} loaded.")

    #here we use the DHAM.
    F_M, MM = DHAM_it(coor_xy_digital_ravelled_total.reshape(prop_index+1, -1, 1), gaussian_params, T=300, lagtime=1, numbins=num_bins, time_tag=time_tag, prop_index=prop_index)
    cur_pos = coor_xy_digital_ravelled_total[-1] #the current position of the particle, in ravelled 1D form.
    
    #determine if the particle has reached the target state.
    end_state_xyz = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]
    end_state_xy = end_state_xyz[:2]
    for index_d, d in enumerate(coor_xy):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state_xy)
        if target_distance < 0.1:
            reach = index_d * config.dcdfreq_mfpt

    return cur_pos, pos_traj, MM, reach, F_M

def get_working_MM(M):
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
        return np.sqrt(np.sum((a-b)**2))
    working_x, working_y = np.unravel_index(working_indices, (N,N), order='C')
    working_states = np.stack((working_x, working_y), axis=1)
    final_state = np.unravel_index(final_index, (N,N), order='C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N,N), order='C')
    return closest_index

def DHAM_it(CV, gaussian_params,X, T=300, lagtime=2, numbins=150, prop_index=0, time_tag=None):
    """
    intput:
    CV: the collective variable we are interested in. now it's 2d.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,bx, by,cx,cy)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params, X)
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True)
    return results

def fes_pca():
    # Take out fes from apply_fes
    x = np.linspace(0, 2*np.pi, 100)
    y = np.linspace(0, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    fes = np.zeros_like(X)

    mass = 12.0 * unit.amu
    system = openmm.System() #we initialize the system every
    system.addParticle(mass)
    system.addParticle(mass)
    system, fes = apply_fes(system = system, particle_idx=0, gaussian_param = None, pbc = config.pbc, amp = config.amp, name = "FES", mode=config.fes_mode, plot = True)

    print(fes)
    
    
    top = Topology()
    top.getUnitCellDimensions() == None
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X1", elem, top._chains[0]._residues[0])
    top.addAtom("X2", elem, top._chains[0]._residues[0])

    z1_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
    z1_pot.addParticle(0)
    x2_pot = openmm.CustomExternalForce("1e3 * x^2")
    y2_pot = openmm.CustomExternalForce("1e3 * y^2")
    z2_pot = openmm.CustomExternalForce("1e3 * z^2")
    x2_pot.addParticle(1)
    y2_pot.addParticle(1)
    z2_pot.addParticle(1)
    system.addForce(z1_pot) #on z, large barrier
    system.addForce(x2_pot)
    system.addForce(y2_pot)
    system.addForce(z2_pot)

    #pbc section
    if config.pbc:
        a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
        b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
        c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
        system.setDefaultPeriodicBoxVectors(a,b,c)

    #integrator
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                        1.0/unit.picoseconds, 
                                        0.002*unit.picoseconds)

    #gaussian_params = random_initial_bias_2d(initial_position = config.start_state, num_gaussians = config.num_gaussian)
    gaussian_params = random_initial_bias_Nd(initial_position = config.start_state, num_gaussians = config.num_gaussian, amplitude = 0.01, num_dimensions = 6)

    #we apply the initial gaussian bias (v small) to the system
    #change to 6D version
    system = apply_2D_bias_cov_FC(system, gaussian_params, pbc = config.pbc, name = "BIAS", num_gaussians = config.num_gaussian)

    #create simulation object, this create a context object automatically.
    # when we need to pass a context object, we can pass simulation instead.
    simulation = openmm.app.Simulation(top, system, integrator, config.platform)
    simulation.context.setPositions(config.start_state)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

    simulation.minimizeEnergy()
    if config.pbc:
        simulation.context.setPeriodicBoxVectors(a,b,c)

    num_propagation = int(config.sim_steps/config.propagation_step)
    frame_per_propagation = int(config.propagation_step/config.dcdfreq_mfpt)
    #this stores the digitized, ravelled, x, y coordinates of the particle, for every propagation.
    pos_traj = np.zeros([num_propagation, frame_per_propagation]) #shape: [num_propagation, frame_per_propagation]
    pos_traj_pca = pos_traj.copy()

    

    cur_pos, pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy = prop_pca(
                                                                    simulation = simulation,
                                                                    prop_index = 0,
                                                                    pos_traj = pos_traj,
                                                                    pos_traj_pca = pos_traj_pca
                                                                    # steps=config.propagation_step,
                                                                    # dcdfreq=config.dcdfreq_mfpt,
                                                                    # stepsize=config.stepsize,
                                                                    # num_bins=config.num_bins,
                                                                    # pbc=config.pbc,
                                                                    # time_tag = time_tag,
                                                                    # top=top,
                                                                    # reach=reach,pca={'mode':'pca_transform'}
                                                                    )
    
    print(cur_pos, pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy)

    exit(0)

if __name__ == "__main__":
    # create directories
    directory_paths = ["./trajectory/explore","./figs/explore","./params","./visited_states","./pca_coords"]
    single_pca_transform = True

    # Save all pca coordinates in a list
    coor_xy_list=[]
    closest_list=[]

    for i in directory_paths:
        if not os.path.exists(i):
            os.makedirs(i)
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.getUnitCellDimensions == None
    top.addChain()
    
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X1", elem, top._chains[0]._residues[0])
    top.addAtom("X2", elem, top._chains[0]._residues[0])

    mass = 12.0 * unit.amu

    # fes_pca()

    for i_sim in range(config.num_sim):
    #def simulate_once():
        print("system initializing")
        #print out all the config.
        print("config: ", config.__dict__)
        
        time_tag = time.strftime("%Y%m%d-%H%M%S")

        #print current time tag.
        print("time_tag: ", time_tag)

        system = openmm.System() #we initialize the system every
        system.addParticle(mass)
        system.addParticle(mass)
        #gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt")
        # RESULT OF fes_visulization IS HERE!!!
        system, fes = apply_fes(system = system, particle_idx=0, gaussian_param = None, pbc = config.pbc, amp = config.amp, name = "FES", mode=config.fes_mode, plot = True)
        if False:
            plt.figure()
            plt.imshow(fes, cmap="coolwarm", extent=[0,2 * np.pi, 0,2 * np.pi], vmin=0, vmax=config.amp *12/7 * 4.184, origin="lower")
            plt.colorbar()
            plt.savefig(f"./figs/{time_tag}_initial_2D_fes.png")
            plt.close()
        #np.savetxt("./figs/fes.txt",fes)
        
        # This part of the code is adding barriers
        z1_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
        z1_pot.addParticle(0)
        x2_pot = openmm.CustomExternalForce("1e3 * x^2")
        y2_pot = openmm.CustomExternalForce("1e3 * y^2")
        z2_pot = openmm.CustomExternalForce("1e3 * z^2")
        x2_pot.addParticle(1)
        y2_pot.addParticle(1)
        z2_pot.addParticle(1)
        system.addForce(z1_pot) #on z, large barrier
        system.addForce(x2_pot)
        system.addForce(y2_pot)
        system.addForce(z2_pot)

        #pbc section
        if config.pbc:
            a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
            b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
            c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
            system.setDefaultPeriodicBoxVectors(a,b,c)

        #integrator
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                            1.0/unit.picoseconds, 
                                            0.002*unit.picoseconds)

        num_propagation = int(config.sim_steps/config.propagation_step)
        frame_per_propagation = int(config.propagation_step/config.dcdfreq_mfpt)
        #this stores the digitized, ravelled, x, y coordinates of the particle, for every propagation.
        pos_traj = np.zeros([num_propagation, frame_per_propagation]) #shape: [num_propagation, frame_per_propagation]
        pos_traj_pca = pos_traj.copy()

        x,y = np.meshgrid(np.linspace(0, 2*np.pi, config.num_bins), np.linspace(0, 2*np.pi, config.num_bins))

        #we start propagation.
        #note num_propagation = config.sim_steps/config.propagation_step
        reach = None
        i_prop = 0
        #for i_prop in range(num_propagation):
        while reach is None:
            if i_prop >= num_propagation:
                print("propagation number exceeds num_propagation, break")
                break
            if i_prop == 0:
                print("propagation 0 starting")
                #gaussian_params = random_initial_bias_2d(initial_position = config.start_state, num_gaussians = config.num_gaussian)
                gaussian_params = random_initial_bias_Nd(initial_position = config.start_state, num_gaussians = config.num_gaussian, amplitude = 0.01, num_dimensions = 6)
                

                np.savetxt(f"./params/{time_tag}_gaussian_fes_param_{i_prop}.txt", gaussian_params)
                #we apply the initial gaussian bias (v small) to the system
                #change to 6D version
                system = apply_2D_bias_cov_FC(system, gaussian_params, pbc = config.pbc, name = "BIAS", num_gaussians = config.num_gaussian)
                #system = apply_6D_bias_FC(system, gaussian_params, pbc = config.pbc, name = "BIAS", num_gaussians = config.num_gaussian)

                #create simulation object, this create a context object automatically.
                # when we need to pass a context object, we can pass simulation instead.
                simulation = openmm.app.Simulation(top, system, integrator, config.platform)
                simulation.context.setPositions(config.start_state)
                simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

                simulation.minimizeEnergy()
                if config.pbc:
                    simulation.context.setPeriodicBoxVectors(a,b,c)

                #now we propagate the system, i.e. run the langevin simulation.
                cur_pos, pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy = prop_pca(simulation = simulation,
                                                                    prop_index = i_prop,
                                                                    pos_traj = pos_traj,
                                                                    pos_traj_pca = pos_traj_pca,
                                                                    steps=config.propagation_step,
                                                                    dcdfreq=config.dcdfreq_mfpt,
                                                                    stepsize=config.stepsize,
                                                                    num_bins=config.num_bins,
                                                                    pbc=config.pbc,
                                                                    time_tag = time_tag,
                                                                    top=top,
                                                                    reach=reach,pca={'mode':'pca_transform'}
                                                                    )

                print("The MM shape is:", MM.shape)
                working_MM, working_indices = get_working_MM(MM)
                print(" ----------------------------- Working MM shape is:", working_MM.shape)


                # add new coor to the total coor
                coor_xy_list.append(coor_xy)
                closest_list.append(closest_value_pca)


                closest_coor = []
                for i in range(2):
                    closest_coor.append(np.digitize(closest_value_pca[i], X[i]) - 1)
                closest_coor = np.array(closest_coor)

                closest_index = np.ravel_multi_index(closest_coor, (config.num_bins, config.num_bins), order = 'C')
                closest_index = find_closest_index(working_indices, closest_index,config.num_bins)
                i_prop += 1
            else:

                print(f"propagation number {i_prop} starting")

                #find the most visited state in last propagation.
                last_traj = pos_traj_pca[i_prop-1,:]
                most_visited_state = np.argmax(np.bincount(last_traj.astype(int))) #this is in digitized, ravelled form.

                ######  change the local start index as the start index (the first element of last_traj) changed Feifan Jun18th
                local_start_state = last_traj.astype(int)[0]

                print("The working mm shape is",working_MM.shape)
                gaussian_params_pca, mean_vec, cov_matrix = try_and_optim_M(working_MM, X = (X[0],X[1]),
                                                working_indices = working_indices,
                                                prop_index = i_prop,
                                                N = config.num_bins,
                                                num_gaussian = config.num_gaussian,
                                                start_index = local_start_state,
                                                end_index = closest_index,
                                                plot = False,
                                                )
                original_means = pca_transform(mean_vec,pca_fitting_file_path = f"./pca_coords/{time_tag}_pca_{0 if single_pca_transform else (i_prop - 1)}.pkl" ,mode = "from_pca")
                original_cov_matrix = []
                pca_cov_matrix_flattened = []
                for i in range(config.num_gaussian):
                    original_space_cov_mat = pca_transform(cov_matrix[i], pca_fitting_file_path = f"./pca_coords/{time_tag}_pca_{0 if single_pca_transform else (i_prop - 1)}.pkl", mode = "matrix_from_pca")
                    original_cov_matrix.append(original_space_cov_mat.flatten())
                    pca_cov_matrix_flattened.append(cov_matrix[i].flatten())
                original_cov_matrix = np.array(original_cov_matrix)
                pca_cov_matrix_flattened = np.array(pca_cov_matrix_flattened)
                print("orignal_space_cov_mat")
                gaussian_params = np.concatenate([gaussian_params_pca[:config.num_gaussian].reshape(config.num_gaussian,1), original_means, original_cov_matrix], axis = 1).T.flatten()
                gaussian_params_pca_full = np.concatenate([gaussian_params_pca[:config.num_gaussian].reshape(config.num_gaussian,1), mean_vec, pca_cov_matrix_flattened], axis = 1).T.flatten()
                #save the gaussian_params
                np.savetxt(f"./params/{time_tag}_gaussian_fes_param_{i_prop}.txt", gaussian_params)
                np.savetxt(f"./params/{time_tag}_gaussian_fes_param_pca_{i_prop}.txt", gaussian_params_pca_full)

                
                ############ transfer to original space ############
                

                #apply the gaussian_params to openmm system.
                simulation = update_bias_2D_FC(simulation = simulation,
                                        gaussian_param = gaussian_params,
                                        name = "BIAS",
                                        num_gaussians=config.num_gaussian,
                                        )
                # simulation = update_bias_6D_FC(simulation = simulation,
                #                         gaussian_param = gaussian_params,
                #                         name = "BIAS",
                #                         num_gaussians=config.num_gaussian,
                #                         )
                
                #we propagate system again
                cur_pos, pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy = prop_pca(simulation = simulation,
                                                                    prop_index = i_prop,
                                                                    pos_traj = pos_traj,
                                                                    pos_traj_pca = pos_traj_pca,
                                                                    steps=config.propagation_step,
                                                                    dcdfreq=config.dcdfreq_mfpt,
                                                                    stepsize=config.stepsize,
                                                                    num_bins=config.num_bins,
                                                                    pbc=config.pbc,
                                                                    time_tag = time_tag,
                                                                    top=top,
                                                                    reach=reach,
                                                                    pca={'mode':'to_pca' if single_pca_transform else 'pca_transform'},
                                                                    )
                
                coor_xy_list.append(coor_xy)
                closest_list.append(closest_value_pca)

                if False:
                        #here we calculate the total bias given the optimized gaussian_params
                        x_total_bias, y_total_bias = np.meshgrid(X[0], X[1]) # shape: [num_bins, num_bins]
                        total_bias = get_total_bias_2d(x_total_bias,y_total_bias, gaussian_params_pca) * 4.184 #convert to kcal/mol
                        plt.figure()
                        plt.imshow(total_bias, cmap="coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin="lower")
                        plt.colorbar()
                        plt.savefig(f"./figs/explore/{time_tag}_total_bias_{i_prop}.png")
                        plt.close()
                        total_bias_big = get_total_bias_2d(X[0], X[1], gaussian_params_pca)* 4.184 #convert to kcal/mol
                        
                        #### original space fes
                        #x1_total_bias, y1_total_bias, z1_total_bias,x2_total_bias, y2_total_bias, z2_total_bias = np.meshgrid(X_orig_dim[0],X_orig_dim[1],X_orig_dim[2],X_orig_dim[3],X_orig_dim[4],X_orig_dim[5]) 
                        #x1_total_bias, y1_total_bias, z1_total_bias,x2_total_bias, y2_total_bias, z2_total_bias = np.meshgrid(np.linspace(0,2*np.pi,config.num_bins),np.linspace(0,2*np.pi,config.num_bins),np.linspace(0,2*np.pi,config.num_bins),np.linspace(0,2*np.pi,config.num_bins),np.linspace(0,2*np.pi,config.num_bins),np.linspace(0,2*np.pi,config.num_bins))
                        #total_bias = get_total_bias_6D((x1_total_bias, y1_total_bias, z1_total_bias,x2_total_bias, y2_total_bias, z2_total_bias),gaussian_params.reshape(43,20).T) * 4.184
                        
                        x_real_total_bias, y_real_total_bias = np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(0,2*np.pi,100))
                        total_bias = get_total_bias_2d(x_real_total_bias,y_real_total_bias, np.concatenate([gaussian_params.reshape(43,20).T[:,0:3],gaussian_params.reshape(43,20).T[:,7:9],gaussian_params.reshape(43,20).T[:,13:15]],axis = 1),no_covariance = False ) * 4.184


                        original_space_fes_integrated = total_bias #reduce_fes_n(total_bias,mode = "uniform_average")
                        plt.figure()
                        plt.imshow(original_space_fes_integrated, cmap = "coolwarm", extent = [0,2*np.pi, 0,2 * np.pi],origin = "lower", interpolation = "gaussian")
                        plt.colorbar()
                        plt.savefig(f"./figs/explore/{time_tag}_total_bias_original_space_{i_prop}.png")
                        plt.close()

                        #here we plot the reconstructed fes from MM.
                        # we also plot the unravelled most_visited_state and closest_index.
                        most_visited_state_unravelled = np.unravel_index(most_visited_state, (config.num_bins, config.num_bins), order='C')
                        closest_index_unravelled = np.unravel_index(closest_index, (config.num_bins, config.num_bins), order='C')
                        plt.figure()
                        plt.imshow(np.reshape(F_M.astype(float),(config.num_bins, config.num_bins), order = "C")*4.184, cmap="coolwarm", extent=[min(X[0]), max(X[0]), min(X[1]), max(X[1])], origin="lower")
                        plt.plot(X[0][most_visited_state_unravelled[0]], X[1][most_visited_state_unravelled[1]], marker='o', markersize=3, color="blue", label = "most visited state (local start)")
                        plt.plot(X[0][closest_index_unravelled[0]], X[1][closest_index_unravelled[1]], marker='o', markersize=3, color="red", label = "closest state (local target)")
                        plt.legend()
                        plt.colorbar()
                        plt.savefig(f"./figs/explore/{time_tag}_reconstructed_fes_{i_prop}.png")
                        plt.close()


                        #we plot here to check the original fes, total_bias and trajectory.
                    
                        #we add the total bias to the fes.
                        #fes += total_bias_big
                        plt.figure()
                        plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7 * 4.184, origin="lower")
                        plt.colorbar()
                        plt.xlabel("x")
                        #plt.xlim([-1, 2*np.pi+1])
                        #plt.ylim([-1, 2*np.pi+1])
                        plt.ylabel("y")
                        plt.title("FES mode = multiwell, pbc=False")
                        
                        #additionally we plot the trajectory.
                        # first we process the pos_traj into x, y coordinate.
                        # we plot all, but highlight the last prop_step points with higher alpha.

                        pos_traj_flat = pos_traj[:i_prop, :].astype(np.int64).squeeze() #note this is digitized and ravelled.
                        x_unravel, y_unravel = np.unravel_index(pos_traj_flat, (config.num_bins, config.num_bins), order='F') #note the traj is temporary ravelled in F order to adapt the DHAM. #shape: [all_frames, 2]
                        
                        pos_traj_flat_last = pos_traj[i_prop:, :].astype(np.int64).squeeze()
                        x_unravel_last, y_unravel_last = np.unravel_index(pos_traj_flat_last, (config.num_bins, config.num_bins), order='F')
                        
                        grid = np.linspace(0, 2*np.pi, config.num_bins)
                        plt.scatter(grid[x_unravel], grid[y_unravel], s=3.5, alpha=0.3, c='black')
                        plt.scatter(grid[x_unravel_last], grid[y_unravel_last], s=3.5, alpha=0.8, c='yellow') 
                        plt.plot(config.start_state_array[0],config.start_state_array[1], marker='o', markersize=3, color="blue", label = "global start")
                        plt.plot(config.end_state_array[0], config.end_state_array[1], marker='o', markersize=3, color="red", label = "global end")
                        plt.legend(loc='best')                                
                        plt.savefig(f"./figs/explore/{time_tag}_fes_traj_{i_prop}.png")
                        plt.close()    



                if True:
                    #plot in pca space.
                    plt.figure()
                    plt.plot(coor_xy_list[i_prop][:,0], coor_xy_list[i_prop][:,1], label=f'traj {i_prop}')
                    plt.plot(closest_list[i_prop][0], closest_list[i_prop][1], marker='o', color='red', label=f'closest point {i_prop}')
                    plt.plot(coor_xy_list[i_prop][0,0], coor_xy_list[i_prop][0,1], marker='o', color='blue', label=f'start point {i_prop}')

                    plt.plot(coor_xy_list[i_prop-1][:,0], coor_xy_list[i_prop-1][:,1], label=f'traj {i_prop-1}')
                    plt.plot(closest_list[i_prop-1][0], closest_list[i_prop-1][1], marker='x', label=f'closest point {i_prop-1}')
                    plt.plot(coor_xy_list[i_prop-1][0,0], coor_xy_list[i_prop-1][0,1], marker='x', label=f'start point {i_prop-1}')


                    plt.title("traj in pca space.")
                    plt.legend()
                    plt.savefig(f"./test_traj_p{i_prop}{i_prop-1}.png")
                    plt.close()



                #update working_MM and working_indices
                working_MM, working_indices = get_working_MM(MM)
                closest_coor = []
                for i in range(2):
                    closest_coor.append(np.digitize(closest_value_pca[i], X[i]) - 1)
                closest_coor = np.array(closest_coor)

                closest_index = np.ravel_multi_index(closest_coor, (config.num_bins, config.num_bins), order = 'C')
                #update closest_index
                closest_index = find_closest_index(working_indices, closest_index, config.num_bins)
                i_prop += 1

        #we have reached target state, thus we record the steps used.
        total_steps = i_prop * config.propagation_step + reach * config.dcdfreq_mfpt
        print("total steps used: ", total_steps)

        with open("./total_steps_mfpt.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])

        #save the pos_traj
        np.savetxt(f"./visited_states/{time_tag}_pos_traj.txt", pos_traj)

    """from multiprocessing import Pool
    
    multi_process_result = []
    for _ in range(config.num_sim//config.NUMBER_OF_PROCESSES):
        with Pool(config.NUMBER_OF_PROCESSES) as p:
            multi_process_result.extend(p.map(simulate_once, range(config.NUMBER_OF_PROCESSES)))
"""
print("all done")
