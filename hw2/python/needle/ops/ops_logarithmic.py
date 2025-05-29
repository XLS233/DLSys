from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=1, keepdims=True)
        Z_exp = array_api.exp(Z - Z_max)
        Z_sum_exp = array_api.sum(Z_exp, axis=1, keepdims=True)
        logZ = array_api.log(Z_sum_exp)
        return Z - Z_max - logZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        logsumexp_Z = logsumexp(Z, axes=(1,))
        softmax = exp(Z - broadcast_to(reshape(logsumexp_Z, (Z.shape[0], 1)), Z.shape))
        sum_out_grad = summation(out_grad, axes=(1,))
        sum_out_grad = reshape(sum_out_grad, (out_grad.shape[0], 1))
        return out_grad - softmax * sum_out_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        expz = array_api.exp(Z - maxz)
        return array_api.log(array_api.sum(expz, axis=self.axes)) + array_api.max(Z, axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes:
            shape = [1] * len(Z.shape)
            j = 0
            for i in range(len(shape)):
                if i not in self.axes:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return grad_new * exp(Z - node_new)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

