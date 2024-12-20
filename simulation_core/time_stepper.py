from ..gpu_utils.cuda_kernels import evolve_field_gpu
import numpy as np

class TimeStepper:
    """
    Handles time evolution of the lattice field using a numerical scheme.
    """

    def __init__(self, lattice_field, time_step_size: float):
        """
        Initialize the TimeStepper with the lattice field and time step size.

        Args:
            lattice_field: Instance of LatticeField to evolve over time
            time_step_size: Time step size for simulation
        """
        self.lattice_field = lattice_field
        self.time_step_size = time_step_size

        # Initialize GPU buffers if not already done
        self.gpu_initialized = False
        self.current_field_gpu = None
        self.previous_field_gpu = None
        self.next_field_gpu = None

    def initialize_gpu_buffers(self):
        """
        Allocate and initialize GPU buffers for the lattice field.
        """
        self.current_field_gpu = np.empty_like(self.lattice_field.current_field)
        self.previous_field_gpu = np.empty_like(self.lattice_field.previous_field)
        self.next_field_gpu = np.empty_like(self.lattice_field.next_field)

        self.gpu_initialized = True

    def step(self) -> None:
        """
        Perform one time step of evolution on the lattice field using the GPU kernel.
        """
        if not self.gpu_initialized:
            self.initialize_gpu_buffers()

        evolve_field_gpu(
            self.lattice_field.current_field,
            self.lattice_field.previous_field,
            self.lattice_field.next_field,
            self.time_step_size
        )

        # Cycle buffers: Advance the time fields
        self.lattice_field.previous_field, self.lattice_field.current_field = \
            self.lattice_field.current_field, self.lattice_field.next_field
