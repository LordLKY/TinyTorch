"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

# MODIFICATION: to avoid confusion, we don't use numpy as backend directly
# import numpy
# # NOTE: we will import numpy as the array_api
# # as the backend for our computations, this line will change in later homeworks
# import numpy as array_api
from .. import backend_ndarray as array_api
from .ops_tuple import make_tuple

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs.data, out_grad * lhs.data


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * node.inputs[0].data ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0].data, node.inputs[1].data
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs = lhs.data
        rhs = rhs.data
        return out_grad / rhs, - out_grad * lhs / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
          assert len(self.axes) == 2, "Axes must be a tuple of length 2"
          return array_api.swapaxes(a, self.axes[0], self.axes[1])
        return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert len(self.shape) == len(a.shape), "Shape mismatch"
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad_shape = self.shape
        input_shape = node.inputs[0].shape
        grad_dim = len(self.shape)
        input_dim = len(input_shape)

        # find broadcast axes
        broadcast_axes = []
        for i in range(0, grad_dim):
          if i >= input_dim:
            broadcast_axes.append(grad_dim - 1 - i)
          elif grad_shape[grad_dim - 1 - i] != input_shape[input_dim - 1 - i]:
            broadcast_axes.append(grad_dim - 1 - i)
        
        # sum on broadcast axes
        out_grad0 = summation(out_grad, tuple(broadcast_axes))

        # potential reshape
        # (3, 1) broadcast to (3, 2), should transform out_grad0 from (3, ) to (3, 1)
        return reshape(out_grad0, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
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

        # broadcast
        return broadcast_to(out_grad0, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.bmm(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input0 = node.inputs[0].data
        input1 = node.inputs[1].data
        grad0 = matmul(out_grad, transpose(node.inputs[1]))
        grad1 = matmul(transpose(node.inputs[0]), out_grad)

        input0_dim = len(input0.shape)
        input1_dim = len(input1.shape)
        grad0_dim = len(grad0.shape)
        grad1_dim = len(grad1.shape)

        if grad0_dim > input0_dim:
          grad0 = summation(grad0, tuple(range(grad0_dim - input0_dim)))
        if grad1_dim > input1_dim:
          grad1 = summation(grad1, tuple(range(grad1_dim - input1_dim)))
        
        return grad0, grad1
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * 1 / node.inputs[0].data)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0].data)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # reshape(out_grad, node.inputs[0].shape)
        return out_grad * (-tanh(node.inputs[0].data) ** 2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple):
        ### BEGIN YOUR SOLUTION
        return array_api.stack(args, self.axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        return array_api.split(A, self.axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.dilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.undilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
    
    def preprocessA(self, A: NDArray, N, H, W, K, Cin) -> NDArray:
        A_pad = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        newH = H + 2 * self.padding
        newW = W + 2 * self.padding
        outH = (newH - K) // self.stride + 1
        outW = (newW - K) // self.stride + 1
        im2col_shape = (N, outH, outW, K, K, Cin)
        im2col_strides = (newH * newW * Cin,
                  newW * Cin * self.stride,
                  Cin * self.stride,
                  newW * Cin,
                  Cin,
                  1)
        A_im2col = A_pad.as_strided(im2col_shape, im2col_strides).compact()
        A_im2col = array_api.reshape(A_im2col, ((N * outH * outW), K * K * Cin))
        return A_im2col

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, Cin)
        # B: (K, K, Cin, Cout)
        # out: (N, outH, outW, Cout)
        assert len(A.shape) == 4 and len(B.shape) == 4, "Input and kernel must be 4D"
        assert A.shape[3] == B.shape[2], "Input channel and kernel channel do not match"
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape
        newH = H + 2 * self.padding
        newW = W + 2 * self.padding
        outH = (newH - K) // self.stride + 1
        outW = (newW - K) // self.stride + 1
        A_im2col = self.preprocessA(A, N, H, W, K, Cin)
        B_im2col = array_api.reshape(B, (K * K * Cin, Cout))
        C = array_api.bmm(A_im2col, B_im2col)
        return array_api.reshape(C, (N, outH, outW, Cout))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs[0].data, node.inputs[1].data
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape
        _, outH, outW, _ = out_grad.shape
        # dA
        B_flip = flip(B, (0, 1)).transpose((2, 3))   
        if self.stride > 1:
            out_grad_dilate = dilate(out_grad, (1, 2), self.stride-1)
        else:
            out_grad_dilate = out_grad
        dA = conv(out_grad_dilate, B_flip, padding=K-1)
        dA = Tensor(dA.cached_data[:, self.padding:H+self.padding, self.padding:W+self.padding, :],
                    device=dA.device,
                    dtype=dA.dtype,
                    requires_grad=False)
        # dB
        A_data = A.realize_cached_data()
        A_im2col = self.preprocessA(A_data, N, H, W, K, Cin)
        A_im2col = Tensor(A_im2col.compact(), device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        out_grad_im2col = reshape(out_grad, (N * outH * outW, Cout))
        dB = matmul(transpose(A_im2col), out_grad_im2col)
        dB = reshape(dB, (K, K, Cin, Cout))

        return dA, dB
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


# Ops below only support forward pass

class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.max(a, axis=self.axes)

    def gradient(self, out_grad, node):
        # tensor with requires_grad==True is not supported
        raise NotImplementedError()


def max(a, axes=None):
    return Max(axes)(a)