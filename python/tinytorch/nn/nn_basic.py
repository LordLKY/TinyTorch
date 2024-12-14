"""The module.
"""
from typing import List, Callable, Any
from tinytorch.autograd import Tensor
from tinytorch import ops
import tinytorch.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
          self.bias = Parameter(init.kaiming_uniform(1, out_features, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if self.bias is not None:
          return ops.add(ops.matmul(X, self.weight), self.bias.broadcast_to((batch_size, self.out_features)))
        return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        original_shape = list(X.shape)
        new_shape = [original_shape[0], 1]
        for i in range(1, len(original_shape)):
          new_shape[1] *= original_shape[i]
        return X.reshape(tuple(new_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_dim, v_dim = logits.shape[0], logits.shape[1]
        y_one_hot = init.one_hot(v_dim, y, device=y.device, dtype=y.dtype) * logits
        logsumexp = ops.logsumexp(logits, axes=(1, ))
        return (logsumexp.sum() - y_one_hot.sum()) / batch_dim
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          batch_dim = x.shape[0]
          x_mean = x.sum(axes=(0, )) / batch_dim
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data
          x_minus_mean = x - x_mean.reshape((1, self.dim)).broadcast_to(x.shape)
          x_var = (x_minus_mean ** 2).sum(axes=(0, )) / batch_dim
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data
          x_std = (x_var + self.eps) ** 0.5
          x_normed = x_minus_mean / x_std.reshape((1, self.dim)).broadcast_to(x.shape)
          return self.weight.broadcast_to(x.shape) * x_normed + self.bias.broadcast_to(x.shape)
        else:
          x_minus_mean = x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
          x_std = (self.running_var + self.eps) ** 0.5
          x_normed = x_minus_mean / x_std.reshape((1, self.dim)).broadcast_to(x.shape)
          return self.weight.broadcast_to(x.shape) * x_normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhwc -> norm(nhwc) -> norm(nchw)
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_dim, v_dim = x.shape[0], x.shape[1]
        x_mean = x.sum(axes=(1, )).reshape((batch_dim, 1)) / v_dim
        x_minus_mean = x - x_mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_dim, 1)) / v_dim + self.eps) ** 0.5
        x_normed = x_minus_mean / x_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * x_normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mask = init.randb(*(x.shape), p=self.p, device=x.device, dtype=x.dtype)
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * (kernel_size ** 2)
        fan_out = out_channels * (kernel_size ** 2)
        bias_bound = 1 / (math.sqrt(fan_in) * kernel_size)
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, fan_out,
                                 shape=(kernel_size, kernel_size, in_channels, out_channels),
                                 device=device, dtype=dtype),
            requires_grad=True)
        self.bias = None
        if bias:
            self.bias = Parameter(
                init.rand(out_channels,
                          low=-bias_bound, high=bias_bound,
                          device=device, dtype=dtype),
                requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 4, "Only support 4D input"
        # NCHW -> NHWC
        x = x.transpose((1, 2)).transpose((2, 3))
        assert x.shape[3] == self.in_channels, "Input channel mismatch"
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size // 2)
        if self.bias is not None:
            out = out + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        # NHWC -> NCHW
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION