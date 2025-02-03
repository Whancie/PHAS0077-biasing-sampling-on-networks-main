import config
import util
import fes_functions
import pca_functions

import numpy as np

import csv

import openmm
from openmm import unit
from openmm.unit.quantity import Quantity
from openmm.app.topology import Topology
from openmm.app.element import Element

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Init
    ## Config file load complete notice
    print(f"Using config file at: {config.__file__}")

    ## Save all pca coordinates in a list !!Might be too big for RAM
    coor_xy_list = []
    closest_list = []

    ## Create directories to store data and figures.
    ## Example usage: paths["params"]
    paths = util.create_data_folders()

    ## Setup the element and topology in question
    element, topology, mass = util.create_topology()

    # Start simulation loop
    for i_sim in range(config.number_of_simulations):
        print(f"--- Running simluation #{i_sim + 1} of {config.number_of_simulations}... ---")

        # Create new system every simulation
        system = openmm.System()
        system.addParticle(mass)
        system.addParticle(mass)

        system, fes = fes_functions.apply_fes(system = system, amp = config.amp, plot = config.plot_fes, plot_path = paths["plots"] + "/fes_visualization.png")

        # Setup periodic box vectors
        pi_nanometers = np.pi * unit.nanometers
        a = unit.Quantity((2 * pi_nanometers, 0 * unit.nanometers, 0 * unit.nanometers))
        b = unit.Quantity((0 * unit.nanometers, 2 * pi_nanometers, 0 * unit.nanometers))
        # atom not moving in z so we set it to 1 nm
        c = unit.Quantity((0 * unit.nanometers, 0 * unit.nanometers, 1 * unit.nanometers))
        system.setDefaultPeriodicBoxVectors(a, b, c)

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

        # Add integrator
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
        
        num_propagation = int(config.sim_steps / config.propagation_step)
        frame_per_propagation = int(config.propagation_step / config.dcdfreq_mfpt)

        # Stores the digitized, ravelled, x, y coordinates of the particle, for every propagation.
        # shape: [num_propagation, frame_per_propagation]
        pos_traj = np.zeros([num_propagation, frame_per_propagation])
        pos_traj_pca = pos_traj.copy()

        x, y = np.meshgrid(np.linspace(0, 2 * np.pi, config.num_bins), np.linspace(0, 2 * np.pi, config.num_bins))

        # Start propagation
        print("-- Initial propagation(#0) starting... --")
        gaussian_params = util.random_initial_bias_Nd(initial_position = config.start_state, num_gaussians = config.num_gaussian)

        # Saving gussian paramaters to file for future reference
        np.savetxt(f"{paths["params"]}/gaussian_fes_param_0.txt", gaussian_params)

        # Aapply the initial gaussian bias (v small) to the system
        # Origional note says: change to 6D version
        system = util.apply_2D_bias_cov_FC(system, gaussian_params, num_gaussians = config.num_gaussian)

        # Create simulation object, this create a context object automatically.
        # when we need to pass a context object, we can pass simulation instead.
        simulation = openmm.app.Simulation(topology, system, integrator, config.platform)
        simulation.context.setPositions(config.start_state)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        simulation.minimizeEnergy()

        # Propagate the system, i.e. run the langevin simulation.
        pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy = pca_functions.prop_pca(
            simulation = simulation, 
            prop_index = 0, 
            pos_traj = pos_traj, 
            pos_traj_pca = pos_traj_pca, 
            coor_xy_list = coor_xy_list,
            paths = paths, 
            pca = {'mode':'pca_transform'}
            )
        
        print("The MM shape is:", MM.shape)
        working_MM, working_indices = util.get_working_MM(MM)
        print("Working MM shape is:", working_MM.shape)

        # Add new coor to the total coor
        coor_xy_list.append(coor_xy)
        closest_list.append(closest_value_pca)

        closest_coor = []
        for i in range(2):
            closest_coor.append(np.digitize(closest_value_pca[i], X[i]) - 1)
        closest_coor = np.array(closest_coor)

        closest_index = np.ravel_multi_index(closest_coor, (config.num_bins, config.num_bins), order = 'C')
        closest_index = util.find_closest_index(working_indices, closest_index,config.num_bins)

        # Setup and propagation #0 complete, starting main loop
        for i_prop in range(1, num_propagation):
            print(f"-- Propagation #{i_prop} of {num_propagation - 1} starting... --")
            #find the most visited state in last propagation.
            last_traj = pos_traj_pca[i_prop-1,:]
            most_visited_state = np.argmax(np.bincount(last_traj.astype(int))) #this is in digitized, ravelled form.

            # Change the local start index as the start index (the first element of last_traj) changed Feifan Jun18th
            local_start_state = last_traj.astype(int)[0]

            print("The working mm shape is",working_MM.shape)
            gaussian_params_pca, mean_vec, cov_matrix = util.try_and_optim_M(
                working_MM, X = (X[0], X[1]),
                working_indices = working_indices,
                prop_index = i_prop,
                N = config.num_bins,
                num_gaussian = config.num_gaussian,
                start_index = local_start_state,
                end_index = closest_index,
                paths = paths,
                )
            original_means = pca_functions.pca_transform(
                mean_vec,
                pca_fitting_file_path = f"{paths["pca_coords"]}/0.pkl",
                mode = "from_pca"
                )
            original_cov_matrix = []
            pca_cov_matrix_flattened = []
            for i in range(config.num_gaussian):
                original_space_cov_mat = pca_functions.pca_transform(
                    cov_matrix[i],
                    pca_fitting_file_path = f"{paths["pca_coords"]}/0.pkl",
                    mode = "matrix_from_pca"
                    )
                original_cov_matrix.append(original_space_cov_mat.flatten())
                pca_cov_matrix_flattened.append(cov_matrix[i].flatten())
            original_cov_matrix = np.array(original_cov_matrix)
            pca_cov_matrix_flattened = np.array(pca_cov_matrix_flattened)
            print("orignal_space_cov_mat")
            gaussian_params = np.concatenate([gaussian_params_pca[:config.num_gaussian].reshape(config.num_gaussian, 1), original_means, original_cov_matrix], axis = 1).T.flatten()
            gaussian_params_pca_full = np.concatenate([gaussian_params_pca[:config.num_gaussian].reshape(config.num_gaussian, 1), mean_vec, pca_cov_matrix_flattened], axis = 1).T.flatten()
            #save the gaussian_params
            np.savetxt(f"{paths['params']}/gaussian_fes_param_{i_prop}.txt", gaussian_params)
            np.savetxt(f"{paths['params']}/gaussian_fes_param_pca_{i_prop}.txt", gaussian_params_pca_full)

            ############ transfer to original space ############

            #apply the gaussian_params to openmm system.
            simulation = util.update_bias_2D_FC(
                simulation = simulation,
                gaussian_param = gaussian_params,
                num_gaussians = config.num_gaussian,
                )
            
            #we propagate system again
            pos_traj, pos_traj_pca, MM, reach, F_M, X, closest_value_pca, coor_xy = pca_functions.prop_pca(
                simulation = simulation,
                prop_index = i_prop,
                pos_traj = pos_traj,
                pos_traj_pca = pos_traj_pca,
                coor_xy_list = coor_xy_list,
                paths = paths,
                i_prop = i_prop,
                pca = {'mode': 'to_pca'},
                )
            
            coor_xy_list.append(coor_xy)
            closest_list.append(closest_value_pca)

            #plot in pca space.
            plt.figure()
            plt.plot(coor_xy_list[i_prop][:, 0], coor_xy_list[i_prop][:, 1], label = f'traj {i_prop}')
            plt.plot(closest_list[i_prop][0], closest_list[i_prop][1], marker = 'o', color = 'red', label = f'closest point {i_prop}')
            plt.plot(coor_xy_list[i_prop][0, 0], coor_xy_list[i_prop][0, 1], marker = 'o', color = 'blue', label = f'start point {i_prop}')
            plt.plot(coor_xy_list[i_prop - 1][:, 0], coor_xy_list[i_prop - 1][:, 1], label = f'traj {i_prop - 1}')
            plt.plot(closest_list[i_prop - 1][0], closest_list[i_prop - 1][1], marker = 'x', label = f'closest point {i_prop - 1}')
            plt.plot(coor_xy_list[i_prop - 1][0, 0], coor_xy_list[i_prop - 1][0, 1], marker = 'x', label = f'start point {i_prop - 1}')
            plt.title("traj in pca space.")
            plt.legend()
            plt.savefig(f"{paths['plots']}/test_traj_p{i_prop}{i_prop-1}.png")
            plt.close()

            #update working_MM and working_indices
            working_MM, working_indices = util.get_working_MM(MM)
            closest_coor = []
            for i in range(2):
                closest_coor.append(np.digitize(closest_value_pca[i], X[i]) - 1)
            closest_coor = np.array(closest_coor)

            closest_index = np.ravel_multi_index(closest_coor, (config.num_bins, config.num_bins), order = 'C')
            #update closest_index
            closest_index = util.find_closest_index(working_indices, closest_index, config.num_bins)

        #we have reached target state, thus we record the steps used.
        total_steps = i_prop * config.propagation_step + config.dcdfreq_mfpt
        print("total steps used: ", total_steps)

        with open(f"{paths["root"]}/total_steps_mfpt.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])

        #save the pos_traj
        np.savetxt(f"{paths['root']}/pos_trajectories.txt", pos_traj)

print("----- ALL DONE!!! -----")