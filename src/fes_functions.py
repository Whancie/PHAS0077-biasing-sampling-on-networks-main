import numpy as np
import openmm
import matplotlib.pyplot as plt


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

    # Periodic Gaussian wells
    num_wells = 9
    A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1]
    y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
    sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

    # Add periodic Gaussian potentials to the system
    for i in range(num_wells):
        energy = (
            f"-A{i}*exp(-(periodicdistance(x, x0{i}, {2 * pi})^2)/(2*sigma_x{i}^2) "
            f"- (periodicdistance(y, y0{i}, {2 * pi})^2)/(2*sigma_y{i}^2))"
        )
        force = openmm.CustomExternalForce(energy)
        force.addGlobalParameter(f"A{i}", A_i[i] * 4.184)  # Convert kcal to kJ
        force.addGlobalParameter(f"x0{i}", x0_i[i])
        force.addGlobalParameter(f"y0{i}", y0_i[i])
        force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
        force.addGlobalParameter(f"sigma_y{i}", sigma_y_i[i])
        force.addPerParticleParameter("x")
        force.addPerParticleParameter("y")
        force.usesPeriodicBoundaryConditions()
        force.addParticle(particle_idx, [])
        system.addForce(force)

    # Plot FES if required
    fes = None
    if plot:
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(0, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Add Gaussian wells
        for i in range(num_wells):
            Z -= A_i[i] * np.exp(
                -(np.minimum(abs(X - x0_i[i]), 2 * pi - abs(X - x0_i[i])) ** 2) / (2 * sigma_x_i[i] ** 2)
                - (np.minimum(abs(Y - y0_i[i]), 2 * pi - abs(Y - y0_i[i])) ** 2) / (2 * sigma_y_i[i] ** 2)
            )

        # Plot the periodic FES
        plt.figure()
        plt.imshow(Z, cmap="coolwarm", extent=[0, 2 * np.pi, 0, 2 * np.pi], origin="lower")
        plt.colorbar(label="Free Energy (kJ/mol)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Periodic Free Energy Surface")
        plt.savefig(plot_path)
        plt.close()
        fes = Z

    return system, fes
