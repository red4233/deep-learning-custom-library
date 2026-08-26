import numpy as np
#takes number of neurons and observation and creates a 2d array
# trackable through the index 
class layers:
    # we put the objects in sequential list so that we know what the output layer will be for backpropogation
    object_counter = 0
    object_list = []

    def relu(z):
        return np.maximum(0, z)

    def soft(z):
        prob = np.exp(z)
        soft_max = prob / np.sum(prob)
        return soft_max

    
    
    def __init__ (self, neuron, input_size, activation=None):
        layers.object_counter += 1
        self.number = layers.object_counter
        self.neuron = neuron
        self.input_size = input_size
        self.weights = np.zeros((self.neuron, self.input_size))
        self.bias = np.zeros((self.neuron, 1))
        self.z = []
        self.activation = activation
        layers.object_list.append(layers.object_counter)


    def forward(self, input):
        for i in range(self.neuron):
            x = np.dot(self.weights[i], input) + self.bias[i]
            self.z.append(x)
            # figures out if the object layer uses relu and calcs it. then it returns the value to the global activastion function list of all layers
        if self.activation == "relu":
            self.activation = layers.relu(self.z)
        elif self.activation == "soft":
                self.activation = layers.soft(self.z)
        else:
            self.activation = None


        


