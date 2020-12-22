import numpy as np

##Primera parte, generemos un input que sea una matriz
## Matriz de 2x4
## 0 0
## 0 1
## 1 0
## 1 1
INPUT_X = np.array([[0,0],[0,1],[1,0],[1,1]])
print ("INPUT_X:\n", INPUT_X)
 
##Luego la matriz de los resultados esperados
## en nuestro caso
## 0
## 1
## 1
## 0
EXPECTED_RESULT = np.array([[0,1,1,0]]).T # queremos en formato columna
print ("EXPECTED_RESULT:\n", EXPECTED_RESULT)
## Resultado: la tabla de verdad de la operación XOR
print ("RESULT:\n", np.append(INPUT_X,EXPECTED_RESULT, axis=1))

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
np.random.seed(0)
 

SYN0 = 2*np.random.random((2,1)) - 1
 
# Vamos a preparar iteraciones para aprender
for i in range(20000):

    l0 = INPUT_X

    l1 = sigmoid(np.dot(l0, SYN0))

    l1_error = EXPECTED_RESULT - l1

    l1_delta = l1_error * sigmoid(l1, True)
 
    SYN0 += np.dot(l0.T, l1_delta)
    if (i % 1000) == 0 :
        print ("Error:" + str(np.mean(np.abs(l1_error))))
 
print (l1)