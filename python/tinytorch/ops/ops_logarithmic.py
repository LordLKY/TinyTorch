from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

# import numpy as array_api
from .. import backend_ndarray as array_api

# NOTE: we have SoftmaxLoss layer in nn_basic.py
# class LogSoftmax(TensorOp):
#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# def logsoftmax(a):
#     return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.max_z1 = array_api.max(Z, axis=self.axes, keepdims=True)
        if self.axes is None:
          new_shape = []
          for _ in range(len(Z.shape)):
            new_shape.append(1)
          self.max_z1 = self.max_z1.reshape(tuple(new_shape))
        self.max_z2 = array_api.max(Z, axis=self.axes, keepdims=False)
        Z0 = array_api.sum(array_api.exp(Z - self.max_z1.broadcast_to(Z.shape)), axis=self.axes)
        return array_api.log(Z0) + self.max_z2
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad_shape = out_grad.shape
        input_shape = node.inputs[0].shape
        grad_dim = len(grad_shape)
        input_dim = len(input_shape)

        # recover original shape
        k = 0
        original_shape = []
        real_axes = self.axes
        if self.axes is None:
          real_axes = list(range(input_dim))
        for i in range(0, input_dim):
          if i in real_axes:
            original_shape.append(1)
          else:
            original_shape.append(grad_shape[k])
            k += 1
        
        # reshape to original shape
        out_grad0 = reshape(out_grad, tuple(original_shape))
        node0 = reshape(node, tuple(original_shape))

        out_grad1 = broadcast_to(out_grad0, input_shape)
        node1 = broadcast_to(node0, input_shape)

        # broadcast
        return out_grad1 * exp(node.inputs[0] - node1)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
