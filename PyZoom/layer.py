import numpy as np
from training_data import data
#takes number of neurons and observation and creates a 2d array
# trackable through the index 
class layers:
    # we put the objects in sequential list so that we know what the output layer will be for backpropogation


    
    
    def __init__ (self, neuron, input_size, input=None, activation=None):
        self.neuron = neuron
        self.input_size = input_size
        self.weights = np.zeros((self.neuron, self.input_size))
        self.bias = np.zeros((self.neuron, 1))
        self.z = []
        self.gradient = 0
        self.activation_type = activation
        self.output = activation
        self.input = input
       

   

    def relu(z):
        return np.maximum(0, z)
        
    def soft(z):
        prob = np.exp(z)
        soft_max = prob / np.sum(prob)
        return soft_max


    def forward(self, input):
        for i in range(self.neuron):
            self.z.append(np.array(np.dot(self.weights[i], input) + self.bias[i]))
        
            # figures out if the object layer uses relu and calcs it. then it returns the value to the global activastion function list of all layers
        if self.output == "relu":
            self.output = layers.relu(self.z)
        
        elif self.output == "soft":
                self.output = layers.soft(self.z)

        else:
            self.output = None

        return self.output

        

    def deriv(self, input):
        # makes eah layer derivtive indepedent of the other layers during backpropogation.
        # by seperating final and ongoing derivtives, we can use the final derivtive to update the weights and bias of the layer and the ongoing derivtive to pass back to the previous layer.
        if self.activation_type == "relu":
            self.z = np.array(self.z)
            relu_deriv = np.where(self.z > 0, 1, 0)
            final_deriv = ( relu_deriv @ input.T)
            ongoing_deriv = ( relu_deriv * self.weights)
            
        else:
            soft_deriv = []
            for i in range (len(self.output)):
                if i == data.correct.index(1):
                    x = self.output[i] * (1- self.output[i])
                else:
                    x = self.output[i] * self.output[data.correct.index(1)]
                soft_deriv.append(x)
            soft_deriv = np.array(soft_deriv)
            x = np.where(self.output)
            
            final_deriv = -1 / self.output[data.correct.index(1)] * ((soft_deriv) @ input.T)
            ongoing_deriv = -1 / self.output[data.correct.index(1)] * (self.weights.T @ (soft_deriv))
        
        
        return final_deriv, ongoing_deriv




        


        


