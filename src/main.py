import config
import util
import fes_functions

import openmm
from openmm.unit.quantity import Quantity
from openmm.app.topology import Topology
from openmm.app.element import Element

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
        print(f"Running simluation #{i_sim + 1}...")

        # Create new system every simulation
        system = openmm.System()
        system.addParticle(mass)
        system.addParticle(mass)

        system, fes = fes_functions.apply_fes(system = system, amp = config.amp, plot = config.plot_fes, plot_path = paths["plots"])
        

    # Centralise Data

    # Convariance Matrix

    # Eigenvalue Decomposition

    # Dimention Reduction (Projection)

    # Apply analysis loop

