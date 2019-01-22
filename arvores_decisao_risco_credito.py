# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 14:05:31 2018

@author: Sam Faraday
"""

import pandas as pd

base = pd.read_csv("risco-credito.csv")

previsores = base.iloc[:,0:4].values

classe = base.iloc[:,4].values
#Transformando categóricos em discretos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 

previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

from sklearn.tree import DecisionTreeClassifier , export



classificador = DecisionTreeClassifier(criterion ="entropy")

#treinamento do algoritmo - Construindo a Árvore de Decisão
classificador.fit(previsores, classe)

#imprimir a árvore
print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file = "arvore.dot",
                       feature_names = ["historia", "divida", "garantia", "renda", ],
                       class_names= ["alto", "moderado","baixo"], 
                       filled = True,
                       leaves_parallel = True)

#história boa, dívida alta, garantias nenhuma, renda > 35
#história ruim, dívida alta, garantias adequada, renda < 15
#predict = execução do algoritmo
resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])
print(resultado)

