from enum import Enum, auto

class PrecisionMode(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

class FieldDimension(Enum):
    ONE_D = "1D"
    TWO_D = "2D"
    THREE_D = "3D"

class DenoiserTiming(Enum):
    AFTER_N_STEPS = "after_n_steps"
    END_OF_SIMULATION = "end_of_simulation"

class VisualizationOutputType(Enum):
    IMAGE_SEQUENCE = "image_sequence"
    VOLUME_RENDER = "3d_volume"
