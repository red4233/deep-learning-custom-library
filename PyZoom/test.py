import numpy as np
import pandas as pd


class dat: 
    names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]
    df = pd.read_csv("iris.data", header = None, names = names)
   
    validation = df.loc[:, "sepal_length":"petal_width"].to_numpy()
    
    validation_list = []
    placer = 0
    for i in range (3):
        
        x = validation[placer:placer + 5].reshape(20,)
        placer += 50
        validation_list.append(x)
    validation_list = np.array(validation_list).reshape(15,4)
    print(validation_list)
