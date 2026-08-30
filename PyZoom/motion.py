import numpy as np
from layer import layers 
class net:

    def __init__ (self):
        self.layer_list = []


    def integrate(self, layer):
        self.layer_list.append(layer)

    def forward(self):

        for i in range (len(self.layer_list)):
             if i > 0:
                    output = layers.forward(self.layer_list[i], self.layer_list[i - 1].output )

             else:
                    output = layers.forward(self.layer_list[i], self.layer_list[i].input)

    def backprop(self):
        final_deriv_list = []
        ongoing_deriv_list = []
        deriv_list = []

        for i in range(len(self.layer_list)):
            if i == 0:
                final_deriv, ongoing_deriv = layers.deriv(self.layer_list[i], self.layer_list[i].input)

            else:
                final_deriv, ongoing_deriv = layers.deriv(self.layer_list[i], self.layer_list[i-1].output)
             
            deriv_list.append((ongoing_deriv, final_deriv))
        #deriv_list = np.array(deriv_list)

        for i in range(len(deriv_list)):
              # allows us to split the main derivitive list 
              backprop_map = (deriv_list[::-1][:i + 1:])

              if i == 0:
                   self.layer_list[i].gradient = backprop_map[i][1]
              else:
                   self.layer_list[i].gradient = np.prod(backprop_map[:i, 0]) * backprop_map[i][1]
                   
        

            


#new plan - make 2 diffrent backprop maps one for ongoing deriv and the other for final deriv




      





