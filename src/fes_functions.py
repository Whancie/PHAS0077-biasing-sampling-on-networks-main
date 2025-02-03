import dham

import numpy as np

import openmm

import matplotlib.pyplot as plt

import numpy.typing as nptypes

def apply_fes(system: openmm.System, amp = 7, plot = False, plot_path: str = None):
    """
    Applies a periodic free energy surface (FES) to the OpenMM system and optionally generates a plot of the FES.

    Parameters:
        system (openmm.System): The OpenMM system to which the FES bias will be applied.
        amp (float): Amplitude of the free energy surface (scales the potential).
        plot (bool): Whether to plot the FES.
        plot_path (str): Path to save the FES plot if `plot` is True.

    Returns:
        tuple: Updated system and FES data (as a 2D array if `plot` is True).
    """
    pi = np.pi
    particle_idx = 0  # Applies the potential to the first particle for simplicity

    k = 5  # Steepness of the sigmoid curve
    max_barrier = "1e2"  # Scaling factor for the potential maximum
    offset = 0.7 #the offset of the boundary energy barrier.
    
    # Define boundary potentials using sigmoid functions to create soft walls
    left_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * x - (-{offset}))))")
    right_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (x - (2 * {pi} + {offset})))))")
    bottom_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * y - (-{offset}))))")
    top_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (y - (2 * {pi} + {offset})))))")

    # Apply boundary potentials to the selected particle
    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    bottom_pot.addParticle(particle_idx)
    top_pot.addParticle(particle_idx)

    # Add boundary potentials to the OpenMM system
    system.addForce(left_pot)
    system.addForce(right_pot)
    system.addForce(bottom_pot)
    system.addForce(top_pot)

    # Define periodic Gaussian wells
    num_wells = 9
    num_barrier = 1

    # Parameters for Gaussian energy wells
    A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp  # Amplitudes (kcal/mol)
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1]  # x-coordinates of well centers (nm)
    y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]  # y-coordinates of well centers (nm)
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]  # Width of the wells in x-direction
    sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]  # Width of the wells in y-direction

    # Parameters for barrier peaks (e.g., a diagonal barrier at x = π)
    A_j = np.array([0.3]) * amp
    x0_j = [np.pi]
    y0_j = [np.pi]
    sigma_x_j = [3]
    sigma_y_j = [0.3]

    # Add a flat energy surface before adding Gaussian wells
    energy = str(amp * 4.184) #flat surface
    force = openmm.CustomExternalForce(energy) # Flat potential energy (kJ/mol)
    force.addParticle(particle_idx)
    system.addForce(force)

    # Add Gaussian energy wells to the system
    for i in range(num_wells):
        energy = f"-A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
        force = openmm.CustomExternalForce(energy)

        # Set global parameters for each Gaussian well
        force.addGlobalParameter(f"A{i}", A_i[i] * 4.184) # Convert kcal to kj
        force.addGlobalParameter(f"x0{i}", x0_i[i])
        force.addGlobalParameter(f"y0{i}", y0_i[i])
        force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
        force.addGlobalParameter(f"sigma_y{i}", sigma_y_i[i])

        force.addParticle(particle_idx)
        system.addForce(force)
    
    # Add barrier peaks to the system
    for i in range(num_barrier):
        energy = f"A{i+num_wells}*exp(-(x-x0{i+num_wells})^2/(2*sigma_x{i+num_wells}^2) - (y-y0{i+num_wells})^2/(2*sigma_y{i+num_wells}^2))"
        force = openmm.CustomExternalForce(energy)

        # Set global parameters for the barrier
        force.addGlobalParameter(f"A{i+num_wells}", A_j[i])
        force.addGlobalParameter(f"x0{i+num_wells}", x0_j[i])
        force.addGlobalParameter(f"y0{i+num_wells}", y0_j[i])
        force.addGlobalParameter(f"sigma_x{i+num_wells}", sigma_x_j[i])
        force.addGlobalParameter(f"sigma_y{i+num_wells}", sigma_y_j[i])

        force.addParticle(particle_idx)
        system.addForce(force)

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.linspace(0, 2 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    Z += amp * 4.184 #flat surface

    # Add energy contributions from Gaussian wells
    for i in range(num_wells):
        Z -= A_i[i] * np.exp(-(X - x0_i[i]) ** 2 / (2 * sigma_x_i[i] ** 2) - (Y - y0_i[i]) ** 2 / (2 * sigma_y_i[i] ** 2))

    # Add energy contributions from barrier peaks
    for i in range(num_barrier):
        Z += A_j[i] * np.exp(-(X - x0_j[i]) ** 2 / (2 * sigma_x_j[i] ** 2) - (Y - y0_j[i]) ** 2 / (2 * sigma_y_j[i] ** 2))
    
    # Add boundary energy barriers using sigmoid functions
    total_energy_barrier = np.zeros_like(X)
    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * pi + offset))))) #right
    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * pi + offset)))))
    Z += total_energy_barrier
    
    # If plotting is enabled, generate the free energy surface (FES) plot
    if plot:
        plt.figure()
        plt.imshow(Z, cmap = "coolwarm", extent = [0, 2 * np.pi,0, 2 * np.pi], vmin = 0, vmax=amp * 12 / 7 * 4.184, origin="lower")
        plt.xlabel("x")
        plt.xlim([-1, 1 + 2 * np.pi])
        plt.ylim([-1, 1 + 2 * np.pi])
        plt.ylabel("y")
        plt.title("FES mode = multiwell, pbc=False")
        plt.colorbar()
        plt.savefig(plot_path)
        plt.close()
        fes = Z

    return system, fes

def DHAM_it(CV: nptypes.NDArray, gaussian_params: list, X: nptypes.NDArray, T = 300, lagtime = 2, numbins = 150, prop_index = 0, paths: dict = None):
    """
    Paramaters:
    ---
    CV: ndarray
        The collective variable we are interested in. now it's 2d.
    gaussian_params: list
        The parameters of bias potential. (in our case the 10-gaussian params)
        Format: (a,bx, by,cx,cy)
    T: int
        temperature 300

    Return:
    ---
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = dham.DHAM(gaussian_params, X)
    d.setup(CV, T, prop_index = prop_index)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True)
    return results