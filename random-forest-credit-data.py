# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 14:05:31 2018

@author: Sam Faraday
"""
import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
#Analisa os Missing Values com a estratégia de média (mean)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Substitui os valores mssing
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])


#escalonando / nivelando as variáveis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Dividindo a base de testes e treinamento
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#n_estimators determina a quantidade de áravores na floresta
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)
# importação da biblioteca
# criação do classificador
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)