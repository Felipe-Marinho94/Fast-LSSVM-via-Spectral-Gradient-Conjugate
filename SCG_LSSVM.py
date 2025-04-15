'''
Implementação de classe para os métodos fit e predict
para a abordagem SCG_LSSVM
Data:16/11/2024
'''

#---------------------------------------------------------------
#Carregando alguns pacotes relevantes
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg, dot
from Utils import Utils
from Kernel import Kernel

#---------------------------------------------------------------
#Implementando a classe
#---------------------------------------------------------------
class SCG_LSSVM(Kernel):

    #Método Construtor
    def __init__(self, gamma = 0.5, ni = 2, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.ni = ni
    
    #---------------------------------------------------------------
    #Função para ajustar o modelo LSSVM utilizando um novo método dos
    #gradientes conjugados espectrais
    #SOURCE: https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-020-02375-z
    #---------------------------------------------------------------
    def fit(self, X, y, epsilon = 0.001, N = 100):
        '''
        Interface do método
        Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
        a metodologia SCFGSMO, retornando os multiplicadores de lagrange
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
        Gradiente = Utils.gradient(alphas, K_tilde, y)

        #Direção conjugada inicial
        s = -Gradiente

        #iteração
        k = 0

        #erro
        erro = []

        #Loop de iteração
        while k <= N:

            #Critério de parada
            if linalg.norm(Gradiente) <= epsilon:
                break

            #cálculo do parâmetro rho usando busca linear de Wolfe
            rho = -dot(s, Gradiente)/dot(dot(s, K_tilde), s)

            #Atualização
            alphas_anterior = alphas
            Gradiente_anterior = Gradiente
            alphas = alphas + rho * s
            Gradiente = Utils.gradient(alphas, K_tilde, y)

            #Armazenando o erro
            erro.append(linalg.norm(Gradiente))

            #calculando os parâmetros theta e beta
            ##Calculando o parâmetro phi
            d_anterior = alphas - alphas_anterior
            l_anterior = Gradiente - Gradiente_anterior

            parameter_phi = Utils.phi(d_anterior, l_anterior)
            parameter_phi_barra = Utils.phi_barra(d_anterior, l_anterior)
            parameter_p = Utils.p(Gradiente, d_anterior, l_anterior)
            parameter_rho_otimo = Utils.rho_otimo(d_anterior, Gradiente_anterior, l_anterior, epsilon, parameter_p)
            parameter_beta_DY = Utils.beta_DY(Gradiente, d_anterior, l_anterior)

            theta = max(min(parameter_rho_otimo, parameter_phi_barra), parameter_phi)
            beta = theta * parameter_beta_DY

            #Atualização
            s = (-theta * Gradiente) + (beta * d_anterior)

            #Incremento
            k = k + 1
        
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
            estimado[i] = (np.dot(np.squeeze(alphas), K[i]))
        
        return estimado