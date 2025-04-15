'''
Implementação de classe para os métodos fit e predict
para o método de estado da arte CFGSMO_LSSVM
Data:16/11/2024
'''

#---------------------------------------------------------------
#Carregando alguns pacotes relevantes
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Kernel import Kernel
from Utils import Utils

#---------------------------------------------------------------
#Implementando a classe
#---------------------------------------------------------------
class CFGSMO_LSSVM(Kernel):

    #Método construtor
    def __init__(self, gamma = 0.5, ni = 2, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.ni = ni
    
    #---------------------------------------------------------------
    #Função para ajustar o modelo LSSVM utilizando o método CFGSMO
    #SOURCE: https://link.springer.com/article/10.1007/s00521-022-07875-1
    #---------------------------------------------------------------
    def fit(self, X, y, epsilon = 0.001, N = 100):
        '''
        Interface do método
        Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
        a metodologia CFGSMO, retornando os multiplicadores de lagrange
        ótimos para o problema de otimização.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        gamma: termo de regularização de Tichonov (escalar)
        kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
        epsilon: Tolerância (Critério de parada)
        N: Número máximo de iterações

        OUTPUT:
        vetor ótimo de multiplicadores de Lagrange estimados.
        '''

        #Construindo a matriz de kernel
        n_samples, n_features = X.shape
        
        #Matriz de Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                
                #Kernel trick
                if self.kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X[i], X[j])
                
                if self.kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(self.gamma, X[i], X[j])
                
                if self.kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X[i], X[j])
                
        
        #Regularização de Tichonov
        K_tilde = K + (1/self.ni)*np.diag(np.full(K.shape[0], 1))
        
        #Inicialização
        #Multiplicadores de Lagrange
        alphas = np.zeros(n_samples)

        #Gradiente
        Gradient = -y

        #direções conjugadas
        s = np.zeros(n_samples)
        t = np.zeros(n_samples)

        #Termo tau
        tau = 1

        #Controle de iteração
        k = 0

        #erro
        erro = []

        #Laço de iteração
        while k <= N:

            #incremento
            k = k + 1

            #Seleção do par de multiplicadores de Lagrange para atualização
            i, j = Utils.FGWSS(Gradient, K_tilde)

            #Realizando as atualizações
            r = (t[j] - t[i])/tau
            s = np.eye(1, n_samples, i)[0] - np.eye(1, n_samples, j)[0] + r * s
            t = K_tilde[:, i] - K_tilde[:, j] + r * t
            tau = t[i] - t[j]

            #Calculando o parâmetro rho
            rho = (Gradient[j] - Gradient[i])/tau

            #Atualização da variável Dual
            alphas = alphas + rho * s

            #Atualização do gradiente
            Gradient = Gradient + rho * t

            #Armazenando o erro
            erro.append(np.max(Gradient) - np.min(Gradient))

            #Condição de parada
            if np.abs(np.max(Gradient) - np.min(Gradient)) <= epsilon:
                break
        
        #Resultados
        resultados = {"mult_lagrange": alphas,
                    "erro": erro}
        
        return resultados
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para a primeira proposta considerando um
    #problema de classificação
    #------------------------------------------------------------------------------
    def predict(self, alphas, X_treino, X_teste):
        #Inicialização
        estimado = np.zeros(X_teste.shape[0])
        n_samples_treino = X_treino.shape[0]
        n_samples_teste = X_teste.shape[0]
        K = np.zeros((n_samples_teste, n_samples_treino))
        
        #Construção da matriz de Kernel
        for i in range(n_samples_teste):
            for j in range(n_samples_treino):
                
                if self.kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X_teste[i], X_treino[j])
                
                if self.kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X_teste[i], X_treino[j])
                
                if self.kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(self.gamma, X_teste[i], X_treino[j])
                
            #Realização da predição
            estimado[i] = np.sign(np.dot(np.squeeze(alphas), K[i]))
        
        return estimado
