import numpy as np
from layer import layers


# Stores layers in forward order and coordinates their forward and backward passes.
class net:

    def __init__ (self):
        self.layer_list = []


    def integrate(self, layer):
        self.layer_list.append(layer)

    def forward(self, target):
        # Each layer consumes the activation produced by the preceding layer.
       
        for i in range (len(self.layer_list)):
             self.layer_list[i].z = []
             if i > 0:
                    output = layers.forward(self.layer_list[i], self.layer_list[i - 1].output)

             else:
                    output = layers.forward(self.layer_list[i], self.layer_list[i].input)
        

             if i == len(self.layer_list)-1:
                          loss = layers.loss(output, target)
                          highest_prediction = layers.accuracy(output, target)
            

        return loss, highest_prediction

        
        

    
    def backprop(self, target):
        # Keep the two kinds of local derivatives separate so they can be
        # combined for each layer after every layer has completed its forward pass.
        final_deriv_list = []
        ongoing_deriv_list = []


        

        for i in range(len(self.layer_list)):
            if i == 0:
                final_deriv, ongoing_deriv = layers.deriv(self.layer_list[i], self.layer_list[i].input)

            else:
                final_deriv, ongoing_deriv = layers.deriv(self.layer_list[i], self.layer_list[i-1].output, target)
              
           
            final_deriv_list.append(final_deriv)
            ongoing_deriv_list.append(ongoing_deriv)

        # Reverse the layer objects so index 0 is the output layer during backpropagation.
        self.layer_list = (self.layer_list[::-1])

        for i in range(len(final_deriv_list)):
              # Select the output-to-current-layer section of each derivative list.
              final_map = (final_deriv_list[::-1][:i + 1:])
              ongoing_map = (ongoing_deriv_list[::-1][:i + 1:])
              matrix_multi_ongoing = ongoing_map[0]
              # Multiply the derivative matrices between the output and the
              # current layer. The current layer's final_deriv supplies the
              # remaining local derivative with respect to its weights.
              for j in range (1, len(ongoing_map) - 1):
                   matrix_multi_ongoing = matrix_multi_ongoing @ ongoing_map[j]
                  
              # Combine the separate output-neuron contributions into the
              # gradient arriving at the current layer.
              matrix_multi_ongoing = np.sum(matrix_multi_ongoing, axis = 0).reshape(-1,1) 

              if i == 0:
                   self.layer_list[i].gradient += final_map[i]
              else:
                   self.layer_list[i].gradient += matrix_multi_ongoing * final_map[i]

        

        self.layer_list = (self.layer_list[::-1])

    def update_weights(self, batch_size):
            
            for i in self.layer_list:
                i.gradient =  i.gradient / batch_size
                i.weights = i.weights - (layers.learning_rate * i.gradient)
                i.gradient = np.zeros(i.weights.shape)
            
             
            

      
                   
        

            







      




