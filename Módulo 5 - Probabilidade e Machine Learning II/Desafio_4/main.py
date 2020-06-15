#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


from IPython.core.pylabtools import figsize

figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui
# #### Dataset _athletes_ que será utilizado:

# In[5]:


# Sua análise começa aqui.
df = athletes
print("Formato do dataset: ", df.shape)
df.head()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).
# 
# ### Resposta questão 1:
# - Aceita-se a hipótese nula do teste de Shapiro-Wilk, caso o p-value do teste seja maior que o alpha (0.05) estabelecido.

# In[6]:


amostra_height = get_sample(df, 'height', 3000)
shapiro_height = sct.shapiro(amostra_height)
alpha = 0.05


print("p-value do teste de Shapiro-Wilk:", round(shapiro_height[1], 8))
print("Para 𝛼 =", alpha,
      "rejeita-se a hipotese nula, portanto de acordo com a estatística do teste, pode-se afirmar que os dados não vêm de uma distribuição normal.")


# In[7]:


def q1():
    amostra_height = get_sample(df, 'height', 3000)
    shapiro_height = sct.shapiro(amostra_height)
    alpha = 0.05
    return shapiro_height[1] > alpha


# In[8]:


print("Resposta q1: ", q1())


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[9]:


plt.hist(amostra_height, bins=25)
plt.title("Histograma de amostras da variável altura")
plt.show()


# In[10]:


import statsmodels.api as sm
sm.qqplot(amostra_height, fit=True, line="45");
plt.title("Q-Q plot")
plt.show()


# A partir dos gráficos acima, pode-se ver que os dados possuem uma distribuição próxima da normal teórica. O que indica que o resultado do teste de Shapiro-Wilk pode não indicar a realidade dos dados.

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).
# 
# ### Resposta da questão 2:

# In[11]:


jarque_bera_height = sct.jarque_bera(amostra_height)

print("p-value do teste de Jarque-Bera:", round(jarque_bera_height[1], 4))
print("Para 𝛼 =", alpha,
      "rejeita-se a hipótese nula, portanto de acordo com a estatística do teste não podemos afirmar que os dados possuem distribuição normal.")


# In[12]:


def q2():
    alpha = 0.05
    jarque_bera_height = sct.jarque_bera(amostra_height)
    return jarque_bera_height[1] > alpha


# In[13]:


print("Resposta q2: ", q2())


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# ### Respostas questão 3:
# - Hipótese nula: dados possuem distribuição normal

# In[14]:


alpha = 0.05
amostra_weight = get_sample(df, 'weight', 3000)
k2, p = sct.normaltest(amostra_weight)

print("p-valor =", round(p, 8))
print("p-valor < alpha: ", p < alpha)

if p < alpha:
    print("Para 𝛼 =", alpha,
          "hipótese nula pode ser rejeitada.")
else:
    print("Para 𝛼 =", alpha,
          "hipótese nula pode ser aceita.")


# In[15]:


def q3():
    alpha = 0.05
    amostra_weight = get_sample(df, 'weight', 3000)
    k2, p = sct.normaltest(amostra_weight)
    if p < alpha:
        return False # Dados nao vem de distrib. normal
    else:
        return True # Dados podem ser de distrib. normal


# In[16]:


print("Resposta q3: ", q3())


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[17]:


plt.hist(amostra_weight, bins=25)
plt.title("Histograma de amostras da variável peso")
plt.show()


# In[18]:


plt.boxplot(amostra_weight)
plt.title("Boxplot variável peso")
plt.show()


# Pelos gráficos pode-se fazer as seguintes pontuações:
# - O histograma possui uma distribuição próxima de normal, porém possui uma calda na direita.
# - No boxplot nota-se a presença de muitos outliers. Provavelmente este é o motivo da alteração do formato da curva normal.
# - Como não houve qualquer filtro dos dados, uma hipótese levantada para estes outliers é a presença de atletas que participem de esportes, ou categorias de peso elevado.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).
# 
# ### Resposta questão 4:

# In[19]:


weight_log = np.log(amostra_weight)
k2, p = sct.normaltest(weight_log)
alpha = 0.05

print("p-valor =", round(p, 8))
print("p-valor < alpha: ", p < alpha)

if p < alpha:
    print("Para 𝛼 =", alpha,
          "hipótese nula pode ser rejeitada.")
else:
    print("Para 𝛼 =", alpha,
          "hipótese nula pode ser aceita.")


# In[20]:


def q4():
    # Retorne aqui o resultado da questão 4.
    weight_log = np.log(amostra_weight)
    k2, p = sct.normaltest(weight_log)
    if p < alpha:
        return False
    else:
        return True


# In[21]:


print("Resposta q4: ", q4())


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[22]:


plt.hist(weight_log, bins=25)
plt.title("Histograma de amostras da variável peso em escala log")
plt.show()


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).
# 
# ### Resposta questão 5:
# - Hipótese nula: média entre as amostras são iguais
# - Nível de significância (alpha) = 0.05

# In[23]:


# Criando dataframes
br = df.query("nationality == 'BRA'")
can = df.query("nationality == 'CAN'")
usa = df.query("nationality == 'USA'")

print("Shape BR: ", br.shape)
print("Shape CAN: ", can.shape)
print("Shape USA: ", usa.shape)

print("\nNúmero de amostras são diferentes, portanto deverão ser adequadas ao teste.\n")

# Teste de hipotese
alpha = 0.05
result_test = sct.ttest_ind(br['height'], usa['height'], equal_var=False, nan_policy='omit')

print("p-valor =", round(result_test[1], 8))
if result_test[1] < alpha:
    print("Hipótese nula rejeitada. Distribuições tem médias diferentes para alpha =", alpha)
else:
    print("Hipótese nula aceita. Distribuições tem médias iguais para alpha = ", alpha)


# In[24]:


def q5():
    # Criando dataframes
    br = df.query("nationality == 'BRA'")
    can = df.query("nationality == 'CAN'")
    usa = df.query("nationality == 'USA'")

    # Teste com alpha = 0.05
    alpha = 0.05
    result_test = sct.ttest_ind(br['height'], usa['height'], equal_var=False, nan_policy='omit') 

    if result_test[1] < alpha:
        return False
    else:
        return True


# In[25]:


print("Resposta q5: ", q5())


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).
# 
# ### Resposta questão 6:

# In[26]:


# Teste de hipotese
alpha = 0.05
result_test = sct.ttest_ind(br['height'], can['height'], equal_var=False, nan_policy='omit')

print("p-valor =", round(result_test[1], 8))
if result_test[1] < alpha:
    print("Hipótese nula rejeitada. Distribuições tem médias diferentes para alpha =", alpha)
else:
    print("Hipótese nula aceita. Distribuições tem médias iguais para alpha = ", alpha)


# In[27]:


def q6():
    # Criando dataframes
    br = df.query("nationality == 'BRA'")
    can = df.query("nationality == 'CAN'")
    usa = df.query("nationality == 'USA'")

    # Teste de hipótese
    alpha = 0.05
    result_test = sct.ttest_ind(br['height'], can['height'], equal_var=False, nan_policy='omit')

    if result_test[1] < alpha:
        return False
    else:
        return True


# In[28]:


print("Resposta q6: ", q6())


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.
# 
# ### Resposta questão 7:

# In[29]:


# Teste de hipotese
alpha = 0.05
result_test = sct.ttest_ind(usa['height'], can['height'], equal_var=False, nan_policy='omit')

print("p-valor =", round(result_test[1], 8))
if result_test[1] < alpha:
    print("Hipótese nula rejeitada. Distribuições tem médias diferentes para alpha =", alpha)
else:
    print("Hipótese nula aceita. Distribuições tem médias iguais para alpha = ", alpha)


# In[30]:


def q7():
    alpha = 0.05
    result_test = sct.ttest_ind(usa['height'], can['height'], equal_var=False, nan_policy='omit')
    p_valor = result_test[1]
    
    # Transformando em class float
    resp = np.float32(round(p_valor, 8))
    resp_float = round(resp.item(), 8)

    return resp_float


# In[31]:


print("Resposta q7: ", q7())


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
