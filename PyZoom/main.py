import numpy as np
from layer import layers
from motion import net

layer_container = net()
input = np.array([[0.1],[0.2],[0.3]])
hidden_layer = layers(32,3,input,"relu")
layer_container.integrate(hidden_layer)
output_layer = layers(10,32, None, "soft" )
layer_container.integrate(output_layer)
layer_container.forward()
layer_container.backprop()
print("so far so good")

