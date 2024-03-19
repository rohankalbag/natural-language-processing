import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RecurrentPerceptron :
    '''
    Single Recurrent Perceptron with sigmoid activation
    Each forward pass takes 
    -> input of (time, input_size) dimension, corresponding to a sentence containing (time) chunk tags
    Note that that pos_tags  passed as inputs should be one-hot encoded in a (input_size) dim vector
    -> Input labels of (time,) dimension - one binary label corresponding to each chunk tag

    As of now, this unit uses sigmoid as activation and binary cross-entropy as the loss
    '''
    def __init__(self, input_size=9, seq_length=-1, clip_grads=False, verbose=False) :
        '''
        input_size(default:9) : dimension of the one-hot input vector == number of inputs to the neuron
        seq_length(default:-1) : number of time steps for BPTT. Pass -1 to consider all the entire time history
        clip_grads(default:False) : whether to clip gradients
        '''
        self.input_size = input_size # number of inputs to the neuron
        self.seq_length = seq_length # number of time steps to backprop through
        self.clip_grads = clip_grads
        self.verbose = verbose
        
        self.W = torch.randn((input_size,)) # weights for the inputs
        self.V = torch.randn(1)[0]          # weights for the feedback
    def __call__(self, x) :
        # x : (time, input_size)
        a = {}; h = {}; o = {}
        h[-1] = 0
        for t in range(len(x)) :
            a[t] = self.V*h[t-1] + sum(self.W*x[t])
            o[t] = torch.sigmoid(a[t])  # assuming tanh activation for now
            h[t] = o[t]
        return h, o
    def forward(self, x) :
        return self(x)
    def backward(self, x, h, o, y):
        # backward pass: compute gradients going backwards
        dw, dv, dh= torch.zeros_like(self.W), torch.zeros_like(self.V), 0
        dhnext = torch.zeros_like(h[0])
        niter = len(x) if self.seq_length==-1 else self.seq_length
        for t in reversed(range(niter)):
            do = (o[t]-y[t])/(o[t]*(1-o[t])) # derivative of loss 
            dh += do + dhnext # backprop into h
            da = (1 - h[t]) * h[t] * dh # backprop through sigmoid non-linearity 
            dw += da*x[t]
            dv += da*h[t-1]
            dhnext = self.V*da
            if self.clip_grads : 
                dw = dw.clamp(-5., 5.) 
                dv = dv.clamp(-5., 5.)
                dh = dh.clamp(-5., 5.)
        return dw, dv
    
    def update_model(self, dw, dv, lr) :
        # SGD 
        self.W -= lr*dw
        self.V -= lr*dv
        
    def train_step(self, x, y, lr) :
        # x : (time, input_size)
        # y : (time,)
        h, out = self.forward(x)
        eps = 1e-4 # for numerical stability
        loss = F.binary_cross_entropy(torch.tensor(list(out.values())).float().clamp(eps, 1-eps), y)
        if self.verbose : print(f'{loss=}, output : {list(out.values())[-1].unsqueeze(0).float()}')
        self.losslog.append(loss)
        dw, dv = self.backward(x, h, out, y)
        self.update_model(dw, dv, lr)

    def train_epoch(self, xs, ys, lr) :
        # xs : (num_data_points, time, input_size) 
        # ys : (num_data_points, time)
        for x,y in zip(xs, ys) :
            self.train_step(x, y, lr)
        
    def train(self, xs, ys, lr, nepochs=10) :
        # xs : (num_data_points, time, input_size) 
        # ys : (num_data_points, time)
        self.losslog = []
        for _ in range(nepochs) :
            self.train_epoch(xs, ys, lr)

    def infer(self, x, thresh=0.5) :
        # x : (time, input_size)
        return 1*(torch.tensor(list(self(x)[1].values())) >= thresh)
        