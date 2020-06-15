#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


pd.set_option('display.max_columns', None)


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui
# ### Explorando dataset

# In[5]:


# Funcoes para auxiliar na analise de dataset
from IPython.display import display

def missing_data(df):    
    missing_data = pd.DataFrame({'Tipo': df.dtypes,'Dados faltantes (%)': df.isna().sum() / df.shape[0]})
    return missing_data

def sumary_df(df):
    print("Informações básicas do dataset")
    print("\nFormato:", df.shape)
    display(df.head(5))

    print("\nPercentual de dados faltantes:")
    display(missing_data(df).T)

    print("\nEstatísticas das features:")
    display(df.describe())
    
def converte_float(num, casas_decimais):
    aux = np.float32(round(num, casas_decimais))
    num_float = round(aux.item(), casas_decimais)
    return num_float


# In[6]:


# Carregando dataframe e sumario
df = fifa
sumary_df(df)


# Como o percentual de dados nulos no dataset é baixo (0.002%) as linhas com campos nulos serão eliminadas.

# In[7]:


df.dropna(inplace=True)
print("Formato pós limpeza:", df.shape)
display(missing_data(df).T)


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[8]:


def q1():
    pca = PCA().fit(df)
    evr = pca.explained_variance_ratio_
    resp = converte_float(evr[0], 3)
    return resp


# In[9]:


print("Resposta da questão 1: ", q1())


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[10]:


def q2():
    pca_095 = PCA(n_components=0.95)
    X_reduced = pca_095.fit_transform(df)
    return X_reduced.shape[1]


# In[11]:


print("Resposta da questão 2: ", q2())


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[12]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[17]:


def q3():
    pca = PCA().fit(df)
    result = pca.components_.dot(x).round(3)
    return (result[0], result[1])


# In[18]:


print("Resposta da questão 3: ", q3())


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[24]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
    
def q4():
    # Separando Overall como variável de interesse
    X = df.drop('Overall', 1)
    y = df['Overall']

    # Criando objeto RFE e aplicando método
    rfe = RFE(LinearRegression(), n_features_to_select=5)
    rfe.fit(X,y)

    # Armazenando nome das variáveis selecionadas
    mask = rfe.support_
    result = X.columns[mask]

    return list(result)


# In[25]:


print("Resposta da questão 4: ", q4())


# In[ ]:




