import numpy as np
import pandas as pd
import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense

# conversion des valeurs Yes/No en 1/0
def convertToBinary(data):
    size = len(data)
    binaryData = [0] * size
    for i in range(0,size):
        if data[i] == "Yes":
            binaryData[i] = 1
    return binaryData

# chargement des données provenant des fichiers csv fournis
creationDataset = pd.read_csv('./../../Data/Spam/Spam detection - For model creation.csv', sep=";")
predictionDataset = pd.read_csv('./../../Data/Spam/Spam detection - For prediction.csv')

# Séparation des données en valeurs d'entrée et de sortie (attendues) et conversion en tableau numpy
creationInputValues = np.asarray(creationDataset.drop(columns="GOAL-Spam", axis=1).astype(float))
creationOutputValues = np.asarray(convertToBinary(creationDataset["GOAL-Spam"]))

predictionInputValues = np.asarray(predictionDataset.drop(columns="Spam", axis=1).astype(float))
predictionOutputValues = np.asarray(predictionDataset["Spam"].astype(float))

# Definition du modèle keras et ajout des couches 1 et 2
model = Sequential()
model.add(Dense(2, input_shape=(57,), activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

# Compilation du modèle
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# apprentissage du modèle avec les données de création
model.fit(creationInputValues, creationOutputValues, epochs=10, batch_size=10)

# evaluation du modèle avec les données de prédiction
_, accuracy = model.evaluate(predictionInputValues, predictionOutputValues)
print("Accuracy: %.2f" % (accuracy*100))

## make predictions with the model
#predictions = (model.predict(predictionInputValues) > 0.5).astype(int)
#
## summarize the first 5 cases
#print("Example of the first 5 predictions")
#for i in range(5):
#	print('input %s => %d (expected %d)' % (i, predictions[i], predictionOutputValues[i]))
