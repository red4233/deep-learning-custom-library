
import numpy as np
import pandas as pd
from network import net

class data: 
    names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]
    df = pd.read_csv("iris.data", header = None, names = names)
    species = np.array([["Iris-setosa", "Iris-versicolor", "Iris-virginica"]])
    x_train = df.to_numpy().reshape(150,5)
    validation = df.loc[:, "sepal_length":"species"].to_numpy().reshape(150,5)
    validation_list = []
    placer = 0
# comment this
    for i in range (3):
        x = validation[placer:placer + 5].reshape(25,)
        placer += 50
        validation_list.append(x)
    validation_list = np.array(validation_list).reshape(15,5)
    validate_labels =  validation_list[:,4].reshape(-1, 1)
    y_validate = np.where(validate_labels == species, 1, 0)
    x_validate = validation_list[:,:4].astype(np.float64)
    
    x_list = []
    for i in range (len(x_train)):
       if(x_train[i] == validation_list).all(axis=1).any() == False:
              x_list.append(x_train[i])
    x_list = np.array(x_list)
    x_train = x_list[:,:4].astype(np.float64)
    
    train_labels = x_list[:,4].reshape(-1,1)
    y_train = np.where(train_labels == species, 1, 0)

   
    
    
           

    
        
    
    


   