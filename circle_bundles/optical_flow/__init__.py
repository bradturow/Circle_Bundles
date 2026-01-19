"""
Optical flow utilities (Sintel pipeline support).

This subpackage contains functions for:
- contrast computations on patches/frames
- loading / organizing flow frames
- preprocessing / patch extraction pipelines
- optional visualization helpers
"""


from .contrast import *         
from .flow_frames import *
from .flow_processing import *

__all__ = []  
