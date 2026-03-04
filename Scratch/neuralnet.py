
import random
from .brain import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):
    
    def __init__(self, nin): # nin is the number of inputs to the neuron
        #this will create a list of weights for each input, and a bias term, all initialized to random values between -1 and 1
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self, x):
        # w * x + b
        out = sum((wi*xi for wi, xi in zip(self.w, x)),self.b)
        return out.tanh()
    
    def parameters(self):
        return self.w + [self.b]# this will return a list of all the parameters in the neuron, which are the weights and the bias
    
#class Layer is a collection of neurons, and the output of the layer is the output of each neuron in the layer
class Layer(Module):
    #nout is the number of neurons in the layer, and nin is the number of inputs to each neuron
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
        
    #this call function will take a list of inputs, and return a list of outputs, one for each neuron in the layer
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs # if there's only one neuron, return the output of that neuron, otherwise return a list of outputs for each neuron in the layer
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] # this will return a list of all the parameters in the layer, 
                                                                           # which are the parameters of each neuron in the layer
class MLP(Module):
    #nouts is a list of the number of neurons in each layer, including the output layer, 
    #but not including the input layer, which is given by nin
    def __init__(self, nin, nouts):
        #sz is a list of the number of neurons in each layer, including the input layer, which is given by nin, and the output layer, which is given by nouts
        sz = [nin] + nouts
        #in this we will create a list of layers, where each layer is a Layer object, 
        #and the number of neurons in each layer is given by the corresponding value in sz
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] # this will return a list of all the parameters in the network,
                                                                        # which are the parameters of each layer in the network
        
        
            