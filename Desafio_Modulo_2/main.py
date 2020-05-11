#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ### Overview do dataset

# In[3]:


# Questao 1
print("Formato do dataset: ", black_friday.shape) 
black_friday.head(5)


# #### Quantidade de valores nulos

# In[4]:


missing_data = pd.DataFrame({'Coluna': black_friday.columns,
                    'Tipo': black_friday.dtypes,
                    'Dados faltantes (%)': black_friday.isna().sum() / black_friday.shape[0]})
missing_data


# In[5]:


black_friday.describe()


# ### Tipos de variáveis
# - <b>Variáveis de identificação:</b> `User_ID`, `Product_ID`
# - <b>Variáveis categóricas:</b> `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`,	`Product_Category_1`,	`Product_Category_2`,	`Product_Category_3`
# - <b>Variável numérica:</b> `Purchase`

# #### Valores existentes para cada variável categórica:

# In[6]:


categorical_features = black_friday.loc[0, 'Gender':'Product_Category_3'].index

for i in categorical_features:
    print("\nTipos de categoria da variável " + i)
    print(black_friday[i].value_counts())


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    woman_26_to_35 = black_friday.query("Gender == 'F' & Age == '26-35'")
    return len(woman_26_to_35)
    pass

# Caso deseje contabilizar a quantidade de mulheres unicas de 26 a 25
#woman_26_to_35 = black_friday.query("Gender == 'F' & Age == '26-35'")
#woman_26_to_35 = woman_26_to_35.loc[:,['User_ID', 'Gender']].drop_duplicates()
#len(woman_26_to_35)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    unique_id = black_friday['User_ID'].unique()
    return len(unique_id)
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[10]:


def q4():
    type_of_data = list()
    for i in black_friday.columns:
        type_of_data.append(black_friday[i].dtype)
    
    return len(set(type_of_data))
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():  
    list_missing_data = black_friday.isna().sum()
    percentual_missing = max(list_missing_data) / black_friday.shape[0]
    
    return percentual_missing
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    list_missing_data = black_friday.isna().sum()
    return max(list_missing_data)
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[13]:


def q7():
    return black_friday['Product_Category_3'].mode()[0]
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[14]:


def q8():
    min_column = black_friday['Purchase'].min()
    max_column = black_friday['Purchase'].max()
    purchased_data = black_friday['Purchase']

    normalized_purchased = ((purchased_data - min_column) / (max_column - min_column))
    return normalized_purchased.mean()
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[35]:


def q9():
    min_column = black_friday['Purchase'].min()
    max_column = black_friday['Purchase'].max()
    purchased_data = black_friday['Purchase']

    normalized_purchased = ((purchased_data - min_column) / (max_column - min_column))

    # This normalization uses the mean and standard deviation 
    normalized_purchased_2 = (normalized_purchased - normalized_purchased.mean()) / np.std(normalized_purchased)

    normalized_purchased_2 = pd.DataFrame(normalized_purchased_2)
    normalized_purchased_2 = ((normalized_purchased_2 <= 1) & (normalized_purchased_2 >= -1))
    return normalized_purchased_2['Purchase'].value_counts()[1]
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[16]:


def q10():
    product_c2_null_c3_not_null = black_friday.query("Product_Category_2 == 'NaN' & Product_Category_3 != 'NaN'")
    if(len(product_c2_null_c3_not_null) == 0):
        return True
    else:
        return False
    pass

