import numpy as np
import matplotlib.pyplot as plt
import random
import math


def perceptron(x,w,active):
    somme =0
    for i in range (0, len(x)):
        somme += x[i] * w[i]
    if active ==0 :
        return 1 if somme >=0 else -1
    else :   
        return math.tanh(somme)

def apprentissage(x,yd,active,nbApprentissages):
    w = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
    mdiff =[]
    learning_rate = 0.1
    for it in range (0 , nbApprentissages):
        erreur =0 
        for i in range (0 , len(x[0])):
            # On rajoute 1 comme valeur du bias (multiplication par 1 pour la correction des poids)
            point = [1 , x[0][i] , x[1][i] ]
            prediction = perceptron(point , w , active)
            if prediction != yd[i] : 
                for j in range (0 , len(w)):
                    # modification des poids
                    w[j] = w[j]+(yd[i] - prediction)*learning_rate*point[j]
                
            erreur += math.pow(yd[i]-prediction, 2)
        mdiff.append(erreur)
    return w , mdiff

def affiche_classe(x,clas,K,w):
    t=[np.min(x[0,:]),np.max(x[0,:])]
    z=[(-w[0]-w[1]*np.min(x[0,:]))/w[2],(-w[0]-w[1]*np.max(x[0,:]))/w[2]]
    plt.plot(t,z)
    ind=(clas==-1)
    plt.plot(x[0,ind],x[1,ind],"o")
    ind=(clas==1)
    plt.plot(x[0,ind],x[1,ind],"o")
    plt.show()
    
# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]] #
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]] #
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128)-1,np.ones(128)))
w,mdiff=apprentissage(data,oracle,1 , 100)
plt.plot(mdiff)
plt.show()
affiche_classe(data,oracle,2,w)
