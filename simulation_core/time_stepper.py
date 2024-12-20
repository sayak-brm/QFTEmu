from ..gpu_utils.cuda_kernels import evolve_field_gpu

class TimeStepper:
    """
    Handles time evolution of the lattice field using a numerical scheme.
    """

    def __init__(self, lattice_field: LatticeField, time_step_size: float):
        """
        Initialize the TimeStepper with the lattice field and time step size.

        Args:
            lattice_field: Instance of LatticeField to evolve over time
            time_step_size: Time step size for simulation
        """
        self.lattice_field = lattice_field
        self.time_step_size = time_step_size

    def step(self) -> None:
        """
        Perform one time step of evolution on the lattice field.
        """
        # Call the CUDA kernel for time evolution
        evolve_field_gpu(
            self.lattice_field.current_field,
            self.lattice_field.previous_field,
            self.lattice_field.next_field,
            self.time_step_size
        )
        
        # Cycle buffers
        self.lattice_field.previous_field, self.lattice_field.current_field = \
            self.lattice_field.current_field, self.lattice_field.next_field
