import openmm
from openmm.unit import Quantity
from openmm import Vec3
from openmm import unit

import numpy as np

number_of_simulations = 1 # Default 1
amp = 4 # Default 4
plot_fes = True # Default True
sim_steps = int(10000) # Default 10000
propagation_step = 500 # Default 500
dcdfreq = 500 # Default 500
dcdfreq_mfpt = 1 # Default 1
num_bins = 20 # Used to discretize the traj, and used in the DHAM. Default 20

num_gaussian = 20 # Number of gaussians used to placing the bias. Default 29
#starting state (as in coordinate space, from 0 to 2pi.)
start_state = Quantity(value = [Vec3(5.0, 4.0, 0.0),Vec3(0.0, 0.0, 0.0)], unit = unit.nanometers)
end_state = Quantity(value = [Vec3(1.0, 1.0, 0.0),Vec3(0.0, 0.0, 0.0)], unit = unit.nanometers) #need to change.

# platform = openmm.Platform.getPlatformByName('CUDA')
platform = openmm.Platform.getPlatformByName('CPU')

def convert_6D_pos_array(vecs):
    final_coor_1 = vecs[0]
    final_coor_2 = vecs[1]
    final_coor_full = []
    for i in final_coor_1:
        final_coor_full.append(i)
    for i in final_coor_2:
        final_coor_full.append(i)
    return np.array(final_coor_full)

start_state_array = convert_6D_pos_array(start_state.value_in_unit_system(openmm.unit.md_unit_system)) ## hardcoded for now but identical to start_state
end_state_array = convert_6D_pos_array(end_state.value_in_unit_system(openmm.unit.md_unit_system)) ## hardcoded for now but identical to end_state

cutoff = 20 # Default 20, for MSM
stepsize = 0.002 * unit.picoseconds #equivalent to 2 * unit.femtoseconds 4fs.