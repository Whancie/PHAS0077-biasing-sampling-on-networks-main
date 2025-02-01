import time
import os

from openmm import unit
from openmm.unit.quantity import Quantity
from openmm.app.topology import Topology
from openmm.app.element import Element

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
    print(f"This execuation is using time tag: {time_tag}")
    oserror_msg = "[Error] Cannot create root directory for this run, please check your permission and/or avaliable disk space."

    # Create root folder of this run
    directory_root = "./data/" + time_tag + "/"
    try:
        os.makedirs(directory_root)
    except:
        # Just in case
        raise OSError(oserror_msg)

    # Create other folders for this run
    directory_paths = [directory_root + "params", directory_root + "pca_coords", directory_root + "plots"]

    try:
        for folder_name in directory_paths:
            os.makedirs(folder_name)
    except:
        # Just in case
        raise OSError(oserror_msg)
    
    result = {
        "params": directory_paths[0],
        "pca_coords": directory_paths[1],
        "plots": directory_paths[2]
    }
    print("Folder structure creation complete!")
    return result

def create_topology() -> Element | Topology | Quantity:
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