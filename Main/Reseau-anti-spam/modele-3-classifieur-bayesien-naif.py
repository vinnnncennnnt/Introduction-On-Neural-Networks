import pandas as pd
import math
def calculValMoinsEsperanceCaree(x , esp):
    return (x - esp)**2


# On rajoute 0.1 afin d'eviter une division par zero dans le cas d'une variance ou une esperance nulle
def calculerProba(x , esperance , variance  ): 
    return (1/(math.sqrt(2*math.pi*variance)+0.01))*(math.exp(  (-1/ (2*variance))*( pow( x-esperance ,2))))
if __name__=="__main__":  
    dataFrame = pd.read_csv('./../../Data/Spam/Spam detection - For model creation.csv', sep=";")
    dataFramePrediction = pd.read_csv('./../../Data/Spam/Spam detection - For prediction.csv')
    dataFrame['GOAL-Spam']= dataFrame['GOAL-Spam'].apply(lambda x : 0 if x =='No' else 1)
    modelPrediction = dataFramePrediction['Spam'].reset_index().drop(columns='index')
    modelPrediction=modelPrediction['Spam'].to_list()
    probaDataFrameNoSpam = dataFramePrediction.drop(columns='Spam').astype(float)
    probaDataFrameSpam = probaDataFrameNoSpam.copy()

    # le tableau de prédictions
    finalClassDataFrame = probaDataFrameNoSpam.copy()

    dataFrameNoSpam = dataFrame.loc[dataFrame['GOAL-Spam']==0].copy()
    dataFrameSpam = dataFrame.loc[dataFrame['GOAL-Spam']==1].copy()
    
    # Calcul de la probabilite p(Spam = 1) et p(Spam = 0)
    nombreTotale = dataFrame.shape[0]
    nombreDeNoSpam = dataFrameNoSpam.shape[0]
    nombreDeSpam = dataFrameSpam.shape[0]
    probaClassNonSpam, probaClassSpam = nombreDeNoSpam/ nombreTotale , nombreDeSpam / nombreTotale

    # Création des tableaux pour le calcul d'ésperance et variance
    tableauEsperance = pd.DataFrame(columns=dataFrame.columns[: len(dataFrame.columns)])
    tableauVariance = pd.DataFrame(columns=dataFrame.columns[: len(dataFrame.columns)])
    tableauEsperance.drop(columns='GOAL-Spam' , inplace=True)
    tableauVariance.drop(columns='GOAL-Spam' , inplace=True)

    # Calcul d'ésperance
    tableauEsperance.loc['Spam zero']= dataFrameNoSpam.sum( axis=0)
    tableauEsperance.loc['Spam un']= dataFrameSpam.sum( axis=0)

    tableauEsperance.iloc[0,:] = tableauEsperance.iloc[0,:].apply(lambda x : x/dataFrameNoSpam.shape[0])
    tableauEsperance.iloc[1,:] = tableauEsperance.iloc[1,:].apply(lambda x : x/dataFrameSpam.shape[0])
    tableauEsperance=tableauEsperance.reset_index().drop(columns='index')

    for col in dataFrameNoSpam.columns[1:]:
        dataFrameNoSpam[col] = dataFrameNoSpam[col].apply(calculValMoinsEsperanceCaree , args=(tableauEsperance.loc[0 ,col ],))
        dataFrameSpam[col] = dataFrameSpam[col].apply(calculValMoinsEsperanceCaree , args=(tableauEsperance.loc[1 ,col ],))

    # Calcul de la Variance
    tableauVariance.loc['Spam zero']= dataFrameNoSpam.sum( axis=0)
    tableauVariance.loc['Spam un']= dataFrameSpam.sum( axis=0)
    tableauVariance.iloc[0,:] = tableauVariance.iloc[0,:].apply(lambda x : x/(dataFrameNoSpam.shape[0]-1))
    tableauVariance.iloc[1,:] = tableauVariance.iloc[1,:].apply(lambda x : x/(dataFrameSpam.shape[0]-1))
    tableauVariance=tableauVariance.reset_index().drop(columns='index')

    # Calcul du tableau des probabilités
    for col in probaDataFrameNoSpam:
        probaDataFrameNoSpam[col]= probaDataFrameNoSpam[col].apply(calculerProba , args=(tableauEsperance.loc[0,col],tableauVariance.loc[0,col]))
        probaDataFrameSpam[col]= probaDataFrameSpam[col].apply(calculerProba , args=(tableauEsperance.loc[1,col],tableauVariance.loc[1,col]))
    probaDataFrameSpam['ProbaClasseUn'] = probaClassSpam
    probaDataFrameNoSpam['ProbaClasseZero'] = probaClassNonSpam

    probaDataFrameSpam['ProduitProba'] = probaDataFrameSpam.product(axis=1)
    probaDataFrameNoSpam['ProduitProba'] = probaDataFrameNoSpam.product(axis=1)

    # definition des classe en se basant sur la probabilité la plus élevée
    finalClassDataFrame['Class'] = -1
    for i,val in enumerate(finalClassDataFrame['Class']):
        finalClassDataFrame.loc[i ,'Class'] = 0 if probaDataFrameNoSpam.loc[i , 'ProduitProba'] >  probaDataFrameSpam.loc[i , 'ProduitProba']   else 1

    modelCreation = list(finalClassDataFrame['Class'])

    compteur = 0 

    # Comparaison des résultats / prédications
    for i in range (0 , len(modelCreation)):
        if modelCreation[i] == modelPrediction[i] :
            compteur +=1

    print("Modèle de Prédiction ")
    print ("Nombre Total : ", finalClassDataFrame.shape[0])
    print ("Nombre de Spam : ", finalClassDataFrame.loc[finalClassDataFrame['Class']==1].shape[0])
    print ("Nombre de NoSpam : ", finalClassDataFrame.loc[finalClassDataFrame['Class']==0].shape[0])
    print()
    print("Modèle de Création ")
    print ("Nombre Total : ", dataFramePrediction.shape[0])
    print ("Nombre de Spam : ", dataFramePrediction.loc[dataFramePrediction['Spam']==1].shape[0])
    print ("Nombre de NoSpam : ",  dataFramePrediction.loc[dataFramePrediction['Spam']==0].shape[0])
    print()
    print("Nombre de similarités : ", compteur)
    print("Précision ", "{:.2f}".format( (compteur/len(modelCreation))*100),"%")
