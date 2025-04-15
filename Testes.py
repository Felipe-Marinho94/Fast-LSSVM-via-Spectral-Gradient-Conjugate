'''
Sript para testar as classes desenvolvidas para cada modelo proposto
Data:15/11/2024
'''

#-------------------------------------------------------------------------
#Importando as classes desenvolvidas
#-------------------------------------------------------------------------

from LSSVM import LSSVM
from FSLM_LSSVM import FSLM_LSSVM
from FSLM_LSSVM_improved import FSLM_LSSVM_improved
from LSSVM_ADMM import LSSVM_ADMM
from TCSMO_LSSVM import TCSMO_LSSVM
from PTCSMO_LSSVM import PTCSMO_LSSVM
from CFGSMO_LSSVM import CFGSMO_LSSVM
from SCG_LSSVM import SCG_LSSVM
from P_LSSVM import P_LSSVM
from IP_LSSVM import IP_LSSVM
from RFSLM_LSSVM import RFSLM_LSSVM
from LARS_LM_LSSVM import LARS_LM_LSSVM

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import Lars
import matplotlib.pyplot as plt
from matplotlib import style
from Kernel import Kernel
from Utils import Utils
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lars

#Base sintética pseudoaleatória
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 1000, centers = 2, 
               cluster_std = 2, n_features = 10)
 

plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
X[1]
plt.show()
plt.clf()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

y
X
#instanciamento de um novo objeto
#LSSVM
model_LSSVM = LSSVM(gamma = 0.5, tau = 0.5)

resultados = model_LSSVM.fit(X, y)
alphas =  resultados["mult_lagrange"]
b = resultados["b"]
alphas
b
model_LSSVM.predict(alphas, b, X, y, X)

#FSLM_LSSVM_improved
model_FSLM_LSSVM_improved = FSLM_LSSVM_improved(eps = 1)
resultados = model_FSLM_LSSVM_improved.fit(X, y)
alphas =  resultados["mult_lagrange"]
alphas
model_FSLM_LSSVM_improved.predict(alphas, X, X)

#Plotando os resultados
sns.set_theme(style="white")
index = resultados['Indices_multiplicadores'][0]
index
resultados['Indices_multiplicadores']
cm_bright = ListedColormap(["#CC29F5", "#87DB6E"])
fig = plt.subplot(2, 2, 1, label = 'K = 0')
fig.scatter(x=X[:,0], y=X[:,1], c=y,
          cmap=cm_bright,
          edgecolors="black",
          s=35,
      )

fig.scatter(X[index,0], X[index,1], c='black',
          cmap=cm_bright,
          marker="v",
          edgecolors="blue",
          s=36,
          label='support vectors')

sns.set_theme(style="white")
index = resultados['Indices_multiplicadores'][5]
index
fig1 = plt.subplot(2, 2, 2, label = 'K = 5')
fig1.scatter(x=X[:,0], y=X[:,1], c=y,
          cmap=cm_bright,
          edgecolors="black",
          s=35,
      )
fig1.scatter(X[index,0], X[index,1], c='black',
          cmap=cm_bright,
          marker="v",
          edgecolors="blue",
          s=36,
          label='support vectors')

sns.set_theme(style="white")
index = resultados['Indices_multiplicadores'][10]
index
fig2 = plt.subplot(2, 2, 3, label = 'K = 10')
fig2.scatter(x=X[:,0], y=X[:,1], c=y,
          cmap=cm_bright,
          edgecolors="black",
          s=35,
      )
fig2.scatter(X[index,0], X[index,1], c='black',
          cmap=cm_bright,
          marker="v",
          edgecolors="blue",
          s=36,
          label='support vectors')

sns.set_theme(style="white")
index = resultados['Indices_multiplicadores'][16]
index
fig3 = plt.subplot(2, 2, 4, label = 'K = 16')
fig3.scatter(x=X[:,0], y=X[:,1], c=y,
          cmap=cm_bright,
          edgecolors="black",
          s=35,
      )
fig3.scatter(X[index,0], X[index,1], c='black',
          cmap=cm_bright,
          marker="v",
          edgecolors="blue",
          s=36,
          label='support vectors')

plt.style.use('default')
plt.show()

#LSSVM_ADMM
model_LSSVM_ADMM = LSSVM_ADMM(gamma = 0.5)
resultados = model_LSSVM_ADMM.fit(X, y)
alphas = resultados["mult_lagrange"]
model_LSSVM_ADMM.predict(alphas, X, X)

#TCSMO_LSSVM
model_TCSMO_LSSVM = TCSMO_LSSVM(gamma = 0.5, ni = 2)
resultados = model_TCSMO_LSSVM.fit(X, y, 0.001, 100)
alphas = resultados["mult_lagrange"]
alphas
y_hat = model_TCSMO_LSSVM.predict(alphas, X, X)
from sklearn.metrics import accuracy_score
accuracy_score(y_hat, y)

len(resultados['erro'])
sns.lineplot(resultados, x=range(0,len(resultados['erro'])), y = resultados['erro'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()

model_TCSMO_LSSVM_pruning = PTCSMO_LSSVM()
resultados = model_TCSMO_LSSVM_pruning.fit(X, y)
alphas = resultados["mult_lagrange"]
alphas
y_hat = model_TCSMO_LSSVM_pruning.predict(alphas, X, X)
from sklearn.metrics import accuracy_score
accuracy_score(y_hat, y)

sns.lineplot(resultados, x=range(0,len(resultados['erro_pruning'])), y = resultados['erro_pruning'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()


#CFGSMO_LSSVM
model_CFGSMO_LSSVM = CFGSMO_LSSVM()
resultados = model_CFGSMO_LSSVM.fit(X, y)
alphas = resultados["mult_lagrange"]
alphas
model_CFGSMO_LSSVM.predict(alphas, X, X)

#SGC_LSSVM
model_SCG_LSSVM = SCG_LSSVM()
resultados = model_SCG_LSSVM.fit(X, y)
alphas = resultados["mult_lagrange"]
alphas
model_SCG_LSSVM.predict(alphas, X, X)
resultados_df = pd.DataFrame(resultados['erro'])
resultados_df.columns = ['erro']
resultados_df['iteracao'] = resultados_df.index
resultados_df

erros = pd.DataFrame()
for i in range(5):
    X, y = make_blobs(n_samples = 1000, centers = 2, 
               cluster_std = 2, n_features = 2)
    model_SCG_LSSVM = SCG_LSSVM()
    resultados = model_SCG_LSSVM.fit(X, y)
    erros = pd.concat([erros, pd.DataFrame(resultados['erro'])], axis = 0)

erros
realization =pd.concat([pd.DataFrame(['Realization 1'] * 40),
                        pd.DataFrame(['Realization 2'] * 40),
                        pd.DataFrame(['Realization 3'] * 40),
                        pd.DataFrame(['Realization 4'] * 40),
                        pd.DataFrame(['Realization 5'] * 40)], axis = 0)

iteracao = pd.concat([  pd.Series(range(1, 41)),
                        pd.Series(range(1, 41)),
                        pd.Series(range(1, 41)),
                        pd.Series(range(1, 41)),
                        pd.Series(range(1, 41))], axis = 0)
realization = realization.reset_index()
erros = erros.iloc[:200, 0].reset_index()
iteracao = iteracao.reset_index()

realization
erros
iteracao
resultados_df = pd.concat([erros, realization, iteracao], axis = 1)
resultados_df = resultados_df.drop(columns='index', axis = 1)
resultados_df.columns = ['erro', 'realization', 'iteracao']
resultados_df

#Desenvolvendo alguna plots
sns.set_theme(style="white")
palette = sns.color_palette("mako_r", 5)
sns.lineplot(data = resultados_df, x='iteracao', y = 'erro', style='realization', hue = 'realization',
             markers = True, dashes = False, palette = palette)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=range(0,len(resultados['erro'])), y=resultados['erro'],
                    mode='lines+markers',
                    name='lines+markers'))

fig.show()




#P_LSSVM
model_P_LSSVM = P_LSSVM()
resultados = model_P_LSSVM.fit(X, y)
alphas =  resultados["mult_lagrange"]
alphas
b = resultados["b"]
b
model_P_LSSVM.predict(alphas, b, X, y, X)

#IP_LSSVM
model_IP_LSSVM = IP_LSSVM()
resultados = model_IP_LSSVM.fit(X, y)
alphas =  resultados["mult_lagrange"]
alphas
b = resultados["b"]
b
model_IP_LSSVM.predict(alphas, b, X, y, X)

#RFSLM_LSSVM
model_RFSLM_LSSVM = RFSLM_LSSVM()
resultados = model_RFSLM_LSSVM.fit(X, y)
alphas =  resultados["mult_lagrange"]
alphas
b = resultados["b"]
b
model_RFSLM_LSSVM.predict(alphas, b, X, y, X)

#LARS_LM_LSSVM
model_LARS_LM_LSSVM = LARS_LM_LSSVM()
resultados = model_LARS_LM_LSSVM.fit(X, y)
alphas =  resultados["mult_lagrange"]
alphas
b = resultados["b"]
b
resultados['Intercepto']
y_hat = model_LARS_LM_LSSVM.predict(alphas, b, X, y, X)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred = y_hat, y_true = y)


#Realizando alguns testes
X, y = make_regression(n_samples=500, n_features=2, noise=1, random_state=42)

#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state=42)

model = PTCSMO_LSSVM()
resultado = model.fit(X_treino, y_treino)
alphas = resultado['mult_lagrange']
b = resultado['b']
alphas.shape
alphas
b
len(alphas[alphas != 0])

lssvm_hat = model.predict(alphas, X_treino, X_teste)
lssvm_hat
r2_score(y_teste, lssvm_hat)
np.sqrt(mean_squared_error(y_teste, lssvm_hat))