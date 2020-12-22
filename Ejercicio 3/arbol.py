
#Importamos las librerias
import numpy as np
import pandas as pd
from sklearn import tree


# Cargamos los datos csv del dataset preprocesado
path = ""
datacsv = pd.read_csv(path+"preprocessed_cpu.csv")
header = datacsv.columns
data = np.array(datacsv)


# Obternermos el x e y que utilizaremos para entrenar al arbol de desicion 

x = data[:,0:-1]
y = data[:,-1:len(data)-1]

# Creamos el arbolde desicion
calsificador = tree.DecisionTreeClassifier()
#Etnrenamos al arbolde desicion con nustros datos x,y 
clasificador = calsificador.fit(x,y)



#Hacemos prueba con alguno de los datos
a = clasificador.predict([[-0.30358614, -0.2243162,  -0.32449978, -0.62188173, -0.39685082, -0.16456273]])
print("Clasificaion: ",a[0])
tree.plot_tree(clasificador)