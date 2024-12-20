from typing import Dict, Any

from gpu_utils.lattice_field import LatticeField
from simulation_core.time_stepper import TimeStepper
from config.enums import FieldDimension, PrecisionMode

class QftSimulationManager:
    """
    Manages the QFT simulation lifecycle including initialization,
    time evolution, and denoising operations.
    """
    
    def __init__(self):
        self.lattice_field = None
        self.time_stepper = None
        
    def initialize_simulation(self, config_params: Dict[str, Any]) -> None:
        """
        Initialize simulation with provided configuration parameters.
        
        Args:
            config_params: Dictionary containing simulation parameters
        """
        # Initialize lattice field
        self.lattice_field = LatticeField(config_params["field_dimension"], config_params["precision_mode"])

        # Initialize time stepper
        self.time_stepper = TimeStepper(self.lattice_field, config_params["time_step_size"])
    
    def run_simulation(self, num_steps: int) -> None:
        """
        Execute the main simulation loop for specified number of steps.
        
        Args:
            num_steps: Number of time steps to simulate
        """
        if not self.time_stepper:
            raise ValueError("Simulation not initialized. Call initialize_simulation() first.")

        # Run simulation for specified number of steps
        for i in range(num_steps):
            self.time_stepper.step()