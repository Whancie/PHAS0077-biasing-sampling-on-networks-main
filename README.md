# PHAS0077-biasing-sampling-on-networks

This project uses PCA to convert a 6D state space formed by the combined dimensions of
two pseudo atoms in Cartesian space to a 2D basis constructed by the 2PC eigenvectors. The 2D fes surface was applied to the x and y coordinates of atom 1 and the remaining 4 CVs were placed in a steep harmonic potential.



## Structure

The following is an overview of the project's directory structure:

```plaintext

.
2D_langevin_sim_FC
├── config.py                   # Configuration file with simulation parameters
├── dham.py                     # DHAM (Discrete Harmonic Approximation Method) 
├── MSM.py                      # Markov State Model (MSM) implementation
├── util.py                     # Utility functions
├──langevin_sim_mfpt_opt_pca.py # Main script to run the Langevin dynamics
├── trajectory/                 
│   └── explore/                # Subdirectory for exploration phase trajectories
├── figs/                       
│   └── explore/                # Subdirectory for exploration phase figures 
├── params/                     # Directory for saving Gaussian parameters
├── visited_states/             # Directory for saving visited states during the simulation
├── pca_coords/                 # Directory for saving PCA-transformed coordinates
README.md                       # Project documentation
requirements.txt                # Python dependencies

```

## Usage

Run the main script `langevin_sim_mfpt_opt_pca.py`, the terminal would show the information of the progress propagation, mean first passage time message together with saving the plots of trajectories in PC space and the corresponding applied bias for each propagation.