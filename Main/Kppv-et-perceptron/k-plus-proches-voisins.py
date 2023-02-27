import numpy as np
import matplotlib.pyplot as plt
from math import sqrt , pow

# la fonction calcule la distance Euclidienne entre deux points "des tuples"
def dist (pointA , pointB):
    return sqrt(pow(pointA[0] - pointB[0] , 2) + pow(pointA[1] - pointB[1] , 2))

def kppv(x,appren,oracle,K):

    clas = []
    # tableau de distance entre un point test et les autres points d'apprentissage
    tabDistanceEntrePoint = []
    for i in range (0 , len (x [0])):
        pointTest = (x[0][i] , x[1][i])
        for j in range (0 , len(appren[0])):
            pointAppr = (appren[0][j] , appren[1][j])
            tabDistanceEntrePoint.append(
            {
                'indice point test' : i ,
                'indice point appren' : j ,
                'distance' : dist(pointA=pointTest , pointB= pointAppr)
            }
            )

        tabDistanceEntrePoint.sort(key = lambda x: x['distance'])
        nombreDeClassA , nombreDeClassB = 0,0
        for k in range (0 , K):
            if oracle[tabDistanceEntrePoint[k]['indice point appren']] == 1 :
                nombreDeClassA +=1
            else:
                nombreDeClassB += 1
        tabDistanceEntrePoint.clear()
        clas.append(1 if nombreDeClassA > nombreDeClassB else 0 )

    return np.array(clas)

def affiche_classe(x,clas,K):
    for k in range(0,K):
        ind=(clas==k)
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
oracle=np.concatenate((np.zeros(128),np.ones(128)))
test1=np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2=np.transpose(np.random.multivariate_normal(mean2, cov2,64))
test=np.concatenate((test1,test2), axis=1)

K=3
clas=kppv(test,data,oracle,K)
affiche_classe(test,clas,2)
