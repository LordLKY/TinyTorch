from . import ops
from .ops import *

# MODIFICATION: to avoid confusion, we don't use numpy as backend directly
# from .autograd import Tensor, cpu, all_devices
from .autograd import Tensor

# MODIFICATION: add backend
from .backend_ndarray import cpu, cpu_numpy, cuda, all_devices

from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import data
from . import nn
from . import optim