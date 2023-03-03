import torch
import numpy as np

class PhaseConjugate:
    """
    Simulate the wave propagation through reconstructed 3D RI using multi-slice model.
    By manually changing the depth and the lateral position of the desired focus, we can simulate the correct wavefront
    """
    def __init__(self, phase_3d, x, y, depth, device='cpu', **kwargs):