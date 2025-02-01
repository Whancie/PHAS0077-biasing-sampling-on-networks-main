#this is a config file for langevin_sim series.

import time
from openmm import unit
import openmm
import numpy as np

from openmm.unit import Quantity
from openmm import Vec3

NUMBER_OF_PROCESSES = 4 #must be lesser than num_sim.
num_sim = 1
sim_steps = int(10000) #change to run for a whole day.
sim_steps_unbiased = int(5e7)
pbc = False #True is not implemented, we got problem fitting periodic function to 2D fes.
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 4 #6 #10 #for amp applied on fes. note the gaussian parameters for fes is normalized.

propagation_step = 5000 # 500
stepsize = 0.002 * unit.picoseconds #equivalent to 2 * unit.femtoseconds 4fs.
stepsize_unbias = 0.002 * unit.picoseconds #100 times.
num_bins = 20 #used to discretize the traj, and used in the DHAM.
dcdfreq = 500
dcdfreq_mfpt = 1

qspace_num_bins = 100
qspace_low = 1
qspace_high = 10

qspace = np.linspace(qspace_low, qspace_high, qspace_num_bins)

# platform = openmm.Platform.getPlatformByName('CUDA')
platform = openmm.Platform.getPlatformByName('CPU')

num_gaussian = 20 #number of gaussians used to placing the bias.

# #starting state (as in coordinate space, from 0 to 2pi.)
# start_state = Quantity(value = [Vec3(5.0, 4.0, 0.4),Vec3(3.0,2.0,-0.3)], unit = unit.nanometers)
# end_state = Quantity(value = [Vec3(1.0, 1.5, 0.1),Vec3(4.0,1.0,-0.9)], unit = unit.nanometers) #need to change.

#starting state (as in coordinate space, from 0 to 2pi.)
start_state = Quantity(value = [Vec3(5.0, 4.0, 0.0),Vec3(0.0,0.0,0.0)], unit = unit.nanometers)
end_state = Quantity(value = [Vec3(1.0, 1.0, 0.0),Vec3(0.0,0.0,0.0)], unit = unit.nanometers) #need to change.

cutoff = 20

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

#start_state = Quantity(value = [Vec3(5.0, 4.0, 0.4)], unit = unit.nanometers)
#end_state = Quantity(value = [Vec3(1.0, 1.5, 0.1)], unit = unit.nanometers) 

#here we have 3 pre-defined 2D fes, stored as different functions.
fes_mode = 'gussian' #chose from ['gaussian', 'multiwell', 'funnel']
#fes_param_path = ['./params/gaussian_fes_param.txt', './params/multi_well_fes_param.txt', './params/funnel_fes_param.txt']