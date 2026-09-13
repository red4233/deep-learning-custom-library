
import numpy as np
import pandas as pd
from motion import net

class data:
    # Placeholder one-hot label used to locate the correct class in layer.deriv().
    correct = [0,0,1]
    names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]
    
    df = pd.read_csv("iris.data", header = None, names = names)
    #print(df)
    species = np.array([["Iris-setosa", "Iris-virginica", "Iris-versicolor"]])
    labels = df["species"].to_numpy().reshape(150,1)
    x_train = df.loc[:, "sepal_length":"petal_width"].to_numpy().reshape(150,4)
    x_train = x_train[:145].reshape(145,4)
    y_train = np.where(labels == species, 1, 0)
    validation = df.loc[:, "sepal_length":"petal_width"].to_numpy().reshape(150,4)
    validation = validation[145:].reshape(5,4)
    


   