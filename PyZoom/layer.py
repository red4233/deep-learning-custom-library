import numpy as np


# A dense layer for the library's current single-sample, column-vector design.
# For n neurons receiving m inputs, weights has shape (n, m).
class layers:
    def __init__ (self, neuron, input_size, input=None, activation=None):
        self.neuron = neuron
        self.input_size = input_size
        self.weights = np.random.uniform(-0.1, 0.1, size=(self.neuron, self.input_size))
        self.bias = np.random.uniform(-0.1, 0.1, size=(self.neuron, 1))
        # z caches each neuron's value before its activation is applied.
        self.z = []
        self.gradient = np.zeros((self.neuron, self.input_size))
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
        # Calculate one pre-activation for each neuron in this layer.
        for i in range(self.neuron):
            self.z.append(np.array(np.dot(self.weights[i], input) + self.bias[i]))
        
        # Apply the activation selected when the layer was created.
        if self.activation_type == "relu":
            self.output = layers.relu(self.z)
        
        elif self.activation_type == "soft":
                self.output = layers.soft(self.z)

        else:
            self.output = None

        return self.output # self.output.shape = (20, 1)



    def deriv(self, input, target=None):
        # final_deriv contains this layer's local derivatives with respect to its
        # weights. ongoing_deriv contains the derivatives passed toward the
        # preceding layer. net.backprop() combines both with the later layers.
        if self.activation_type == "relu":
            self.z = np.array(self.z)
            relu_deriv = np.where(self.z > 0, 1, 0)  # relu_deriv.shape = (20, 1)
            final_deriv = ( relu_deriv * input.T) # final_deriv.shape = (20, 3)
            ongoing_deriv = ( relu_deriv * self.weights ) # ongoing_deriv.shape = (20,3) 
            
        else:
            # Compute the derivative of the correct-class softmax probability
            # with respect to every output logit.
            soft_deriv = [] # soft_deriv.shape = (20,1)
            for i in range (len(self.output)):
                if i == np.argmax(target):
                    x = self.output[i] * (1- self.output[i])
                else:
                    x = -self.output[i] * self.output[np.argmax(target)]
                soft_deriv.append(x) 
            soft_deriv = np.array(soft_deriv)
            
            # Multiplying by -1 / p(correct) applies the derivative of the
            # cross-entropy loss, -log(p(correct)).
            final_deriv = -1 / self.output[np.argmax(target)] * (soft_deriv * input.T) # final_deriv.shape = (20,20)
            ongoing_deriv = -1 / self.output[np.argmax(target)] * (self.weights * soft_deriv) # ongoing_deriv = (20,20)
        
        
        return final_deriv, ongoing_deriv

    def loss(output,target):
       x = np.argmax(target)
       loss = -np.log(output[x])
       return loss

    def accuracy(output, target):
        x = np.argmax(output)
        if x == np.argmax(target):
            return 1
        else:
            return 0
        

      
      



      

        


        

