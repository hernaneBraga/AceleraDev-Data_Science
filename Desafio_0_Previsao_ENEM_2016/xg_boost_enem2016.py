# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:25:49 2020

@author: herna
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# =============================
#  Funcoes usadas
# ============================

# Retorna 3 dataframes: candidatos ausentes, presente e ordem de NU_INSCRICAO
def load_data(nome_arquivo, variaveis_usadas, teste):    
    
    if(teste == 1):
        variaveis_usadas.remove('NU_NOTA_MT')
    
    df = pd.read_csv(nome_arquivo, sep=',')
    print("\nTamanho inicial dataframe: {}".format(df.shape))
    
    # Reduzindo tamanho do dataframe
    df = df[variaveis_usadas]
    
    # Separando a ordem de inscricao de acordo com o original
    if(teste == 1):
        df_ordem = df['NU_INSCRICAO']
    else:
        df_ordem = df[['NU_INSCRICAO', 'NU_NOTA_MT']]
    
    
    # Candidatos ausentes/desclassificados
    df_ausente = df[ (df["TP_PRESENCA_LC"] != 1) ] 
    if(teste == 1):
        df_ausente['NU_NOTA_MT'] = np.nan
    else:     
        df_ausente.loc['NU_NOTA_MT'] = np.nan
        df_ausente = df_ausente[['NU_INSCRICAO', 'NU_NOTA_MT']]
        
    
    # Candidatos presentes
    df = df[ (df["TP_PRESENCA_LC"] == 1) ] 
    df = df.drop(columns=['TP_PRESENCA_LC']) #coluna desnecessaria
    
    # Retorno funcao
    if(teste == 1):
        print("\nTamanho final dataframe de candidatos presentes: {}".format(df.shape))
        print("\nTamanho final dataframe de candidatos ausentes: {}".format(df_ausente.shape))
        return df, df_ausente, df_ordem
    else:
        print("\nTamanho final dataframe de candidatos presentes: {}".format(df.shape))
        return df

def converte_questionario(df):
    # Questionario socioeconomico
    dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 18}
    
    # Trata os valores faltantes na questao 27
    df['Q027'] = df['Q027'].fillna(0)
    
    df.loc[:,'Q001':'Q047'] = df.loc[:,'Q001':'Q047'].replace(dic)        
    return df

def gera_score(df, w):
    # Funcao que calcular score
    func = (df['Q001']*w[0]+ df['Q002']*w[1] + df['Q006']*w[2] + 
                df['Q024']*w[3] + df['Q025']*w[4] + df['Q026']*w[5] + 
                df['Q047']*w[6] + df['Q027']*w[7])
    
    df['score'] = func    
    return df

def remove_nulos_na(df, lst_colunas):
    print("\nForma do dataframe antes da funcao: {}".format(df.shape))

    df = df.dropna(subset=lst_colunas)
    df = df[df["NU_NOTA_MT"] != 0]
    
    print("\nForma do dataframe apos a funcao: {}".format(df.shape))
    return df


# ===============================
#  Tratamento de dados de treino
# ==============================

variaveis_usadas = ['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 
                    'NU_NOTA_REDACAO', 'CO_UF_RESIDENCIA', 'NU_IDADE', 
                    'TP_COR_RACA', 'TP_ESCOLA', 'TP_ANO_CONCLUIU', 
                    'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 
                    'Q027', 'Q047', 'TP_PRESENCA_LC', 'NU_NOTA_MT']


arquivo_treino = "testfiles/train.csv"


df_treino = load_data(arquivo_treino, variaveis_usadas, 0)
df_treino = converte_questionario(df_treino)

w = [1,2,5,3,10,2,1,5]
df_treino = gera_score(df_treino, w)
df_treino = remove_nulos_na(df_treino, ['NU_NOTA_CN'])

# Cria variaveis dummy
df_treino = pd.get_dummies(df_treino, columns=['TP_COR_RACA',  'CO_UF_RESIDENCIA'], drop_first=True)

# Reordenando o df para deixar a variavel de interesse no final
aux = df_treino['NU_NOTA_MT'].copy()
df_treino = df_treino.drop(columns= ['NU_NOTA_MT'])
df_treino['NU_NOTA_MT'] = aux



# ===============================
#  Tratamento de dados de teste
# ==============================

arquivo_teste = "testfiles/test.csv"

df_test, df_ausente_test, df_ordem_teste = load_data(arquivo_teste, variaveis_usadas, 1)
df_test = converte_questionario(df_test)

w = [1,2,5,3,10,2,1,5]

df_test.fillna(0, inplace=True)
df_test = gera_score(df_test, w)


# Cria variaveis dummy
df_test = pd.get_dummies(df_test, columns=['TP_COR_RACA',  'CO_UF_RESIDENCIA'], drop_first=True)




# =============================
#  Regressao com XGboost
# ============================

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb


# Separando variaveis de entrada e saida
X, y = df_treino.iloc[:,1:-1], df_treino.iloc[:,-1]

# Convertendo em uma estrutura Dmatrix para o XGBoost
data_dmatrix = xgb.DMatrix(data=X,label=y)


# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Criando objeto do XGboost para regressao
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, 
                learning_rate = 0.3, max_depth = 15, alpha = 10, n_estimators = 10)


# Fit do modelo e predicoes
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

# Erro quadrado medio
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))




# =============================
#  K-fold Cross Validation 
# ============================

# Dicionario de parametros
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 12, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=100, early_stopping_rounds=50, 
                    metrics="rmse", as_pandas=True, seed=123)


print("Média de cross-fold validation: RMSE = ", cv_results['test-rmse-mean'].mean() )


# =====================================
#  Usando modelo nos dados do desafio
# =====================================

X_test = df_test.iloc[:,1:]
resultado = xg_reg.predict(X_test)


# =====================================
#  Contruir a resposta e salvando csv
# =====================================

df_test['NU_NOTA_MT'] = resultado

df_resultado = df_test[['NU_INSCRICAO','NU_NOTA_MT']]
df_ausente_test = df_ausente_test[['NU_INSCRICAO','NU_NOTA_MT']]

frames = [df_ausente_test, df_resultado]
csv_resposta = pd.concat(frames)

csv_resposta = csv_resposta.sort_index()
csv_resposta.to_csv('answer.csv', index=False)


# =======================================================
#  Analisando a arvore criada e features mais importantes
# =======================================================

# Visualizar arvore
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [100, 100]
plt.show()


# Importancia variaveis
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
