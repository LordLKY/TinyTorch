"""Optimization module"""
import tinytorch as ttorch
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION 
        for param in self.params:
          if param.grad is None:
            continue
          grad1 = ttorch.Tensor(param.grad.data + self.weight_decay * param.data, device=param.device, dtype=param.dtype, requires_grad=False)
          if param not in self.u.keys():
            self.u[param] = ttorch.Tensor(np.zeros(param.shape), device=param.device, dtype=param.dtype, requires_grad=False)
          self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad1
          param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
          if param.grad is not None:
            grad1 = ttorch.Tensor(param.grad.data + self.weight_decay * param.data, device=param.device, dtype=param.dtype, requires_grad=False)
            grad_v_1 = grad1 ** 2
            if param not in self.m.keys():
              self.m[param] = 0
            if param not in self.v.keys():
              self.v[param] = 0
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad1
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad_v_1
            grad1_bc = self.m[param] / (1 - self.beta1 ** self.t)
            grad_v_1_bc = self.v[param] / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * grad1_bc / (grad_v_1_bc ** 0.5 + self.eps)
        ### END YOUR SOLUTION
