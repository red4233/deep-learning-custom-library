import numpy as np
from layer import layers
from network import net
from training_data import data
import matplotlib.pyplot as plt

layer_container = net()
# One column represents the single example passed through the network.
epoch = 4000
batch =  6
batch_size = 145
validation_size = 5
epoch_list = []
loss_list = []
accuracy_list = []



# Define the network from its input layer through its output layer.
hidden_layer = layers(20, 4, input, "relu")
layer_container.integrate(hidden_layer)
hidden_layer2 = layers(30, 20, None, "relu")
layer_container.integrate(hidden_layer2)
hid = layers(60, 30, None, "relu")
layer_container.integrate(hid)
output_layer = layers(3, 60, None, "soft" )
layer_container.integrate(output_layer)






for i in range (epoch):
    total_loss = 0
    total_accuracy = 0
    epoch_list.append(i+1)
    for i in range (len(data.x_train)): 
        layer_container.layer_list[0].input = data.x_train[i]
        target = data.y_train[i]
        loss, accuracy = layer_container.forward(target)
        total_loss = total_loss + loss
        total_accuracy = total_accuracy + accuracy
        layer_container.backprop(target)
    average_loss_per_batch = total_loss / batch_size
    average_accuracy = total_accuracy / batch_size
    print(f"Loss:{average_loss_per_batch} Accuracy:{average_accuracy}") 
    accuracy_list.append(total_accuracy/batch_size)
    loss_list.append(average_loss_per_batch)
    layer_container.update_weights(batch_size)

total_accuracy = 0
for i in range (len(data.validation)):
    layer_container.layer_list[0].input = data.validation[i]
    target = data.y_train[i+ len(data.x_train)]
    loss, accuracy = layer_container.forward(target)
    total_accuracy = total_accuracy + accuracy
print(f" validation accuracy: {total_accuracy/validation_size}")
    



plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1.5)
plt.title("Loss Over Epoch")
plt.show()

plt.plot(epoch_list, accuracy_list)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy Over Epoch")
plt.show()




               
