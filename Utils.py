'''
Classe contendo algumas fuções úteis no desenvolvimento de outras classes
Data:15/11/2024
'''
#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from numpy import linalg as ln
from sklearn.cluster import DBSCAN
from Kernel import Kernel
from math import sqrt, log
from numpy import dot

#------------------------------------------------------------------------------
#Implementação da classe
#------------------------------------------------------------------------------
class Utils(Kernel):

    #Método construtor
    def __init__(self, gamma, eps, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.eps = eps
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def CG(A, b, epsilon, N):
        #--------------------------------------------------------------------------
        #Método iterativo para a solução de sistemas lineares Ax = B, com A sendo
        #simétrica e definida positiva
        #INPUTS:
        #A:matriz dos coeficientes de ordem m x n (array)
        #B:vetor de termos independentes m x 1 (array)
        #epsilon:tolerância (escalar)
        #N: Número máximo de iterações
        #OUTPUTS:
        #x*:vetor solucao aproximada n x 1 (array) 
        #--------------------------------------------------------------------------
        
        #Inicialização
        i = 0
        x = np.zeros(A.shape[1])
        r = b - A.dot(x)
        r_anterior = r
        
        while (np.sqrt(np.dot(r, r)) > epsilon) | (i <= N) :
            i += 1
            if i == 1:
                p = r
            else:
                beta = np.dot(r, r)/np.dot(r_anterior, r_anterior)
                p = r + beta * p
            
            lamb = np.dot(r, r)/np.dot(p, A.dot(p))
            x += lamb * p
            r_anterior = r
            r += -lamb * A.dot(p)
            
        return x
    
    @staticmethod
    def CG_conditioned(A, b, x, N, epsilon):
        residual = b - A.dot(x)
        preconditioner = np.linalg.inv(np.linalg.cholesky(A))
        
        z = np.dot(preconditioner, residual)
        d = z
        
        error = np.dot(residual.T, residual)
        
        iteration = 0
        while iteration< N and error> epsilon**2:
            q        = np.dot( A, d )
            a        = np.dot(residual.T, z)/np.dot( d.T, q )
            
            phi      = np.dot( z.T,  residual )
            old_res  = residual
            
            x        = x + a * d
            residual = residual - a * q
            
            z        = np.dot( preconditioner, residual )
            beta     = np.dot(z.T, (residual-old_res))/phi # Polak-Ribiere
            d        = z + beta * d
            
            error    = residual.T.dot(residual)
            
            iteration += 1
        
        if iteration<N:
            print('Precisão alcançada. Iterações:', iteration)
        else:
            print('Convergência não alcançada.')
            
        return x
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta Three-Term Conjugate Like
    #SMO LSSVM algoritmo
    #---------------------------------------------------------------
    @staticmethod
    def FGWSS(gradient, K):
        '''
        Interface do Método
        Ação: Visa selecionar os índices dos dois multiplicadores de
        Lagrange que serão atualizados, por meio da heurística do
        método Functional Gain Working Selection Stategy (FGWSS)

        INPUT:
        gradient: Vetor Gradiente (array n)
        K: Matriz de Kernel (array n x n)

        OUTPUT:
        os dois índices selecionados
        '''
        
         #Primeiro índice
        i = np.argmax(np.absolute(gradient))

        #Segundo índice
        exception = i
        m = np.zeros(gradient.shape[0], dtype=bool)
        m[exception] = True

        numerador = (gradient-gradient[i])**2
        denominador = 2 * (K[i, i] * np.ones(K.shape[0]) + np.diag(K) - K[i, :] - K[:, i])
        quociente = np.ma.array(numerador/denominador, mask=m)
        j = np.argmax(quociente)

        return (i, j)
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta Spectral Conjugate
    #LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def gradient(alphas, K_tilde, y):
        '''
        Calcula o valor do gradiente da função objetivo
        INPUT:
        alphas - Multiplicadores de Lagrange (array n x 1)
        '''

        return(np.dot(K_tilde, alphas) - y)
    
    @staticmethod
    def phi(d_anterior, l_anterior):
        '''
        Calcula o parâmetro phi utilzado nas iterações
        '''

        return(dot(d_anterior, l_anterior)/dot(l_anterior, l_anterior))
    
    @staticmethod
    def rho_otimo(d_anterior, gradiente_anterior, l_anterior, epsilon, p):
        '''
        Calcula o parâmetro rho ótimo utilzado nas iterações
        '''

        return(-dot(d_anterior, gradiente_anterior)/(epsilon * dot(l_anterior, l_anterior) * p))
    
    @staticmethod
    def phi_barra(d_anterior, l_anterior):
        '''
        Calcula o parâmetro rho barra utilzado nas iterações
        '''

        return(dot(d_anterior, d_anterior)/(dot(d_anterior, l_anterior)))
    
    @staticmethod
    def p(gradiente, d_anterior, l_anterior):
        '''
        Calcula o parâmetro p utilzado nas iterações
        '''

        return(1- (dot(gradiente, d_anterior)**2)/(dot(gradiente, gradiente)*dot(d_anterior, d_anterior))+
            (dot(gradiente, l_anterior)/(ln.norm(gradiente)*ln.norm(l_anterior))+
                ln.norm(gradiente)/ln.norm(l_anterior))**2)
    
    @staticmethod
    def beta_DY(gradiente, d_anterior, l_anterior):
        '''
        Calcula o parâmetro beta DY utilzado nas iterações
        '''

        return(dot(gradiente, gradiente)/dot(d_anterior, l_anterior))
    

    #---------------------------------------------------------------
    #Funções auxiliares para a proposta Prunning LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def Construct_A_b(X, y, kernel, tau, gamma):
        """
        Interface do método
        Ação: Este método ecapsular o cálculo da matriz A e vetor b para
        a proposta RFSLM-LSSVM.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
        #tau: Parâmetro de regularização do problema primal do LSSVM.

        OUTPUT:
        dois array's representando a matriz A e b, respectivamente.
        """

        #Construção da matriz A e vetor b e inicialização aleatória
        #do vetor de multiplicadores de Lagrange
        n_samples, n_features = X.shape
        
        #Matriz de Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                
                #Kernel trick
                if kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X.iloc[i, :], X.iloc[j, :])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X.iloc[i, :], X.iloc[j, :])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X.iloc[i, :], X.iloc[j, :])
        
        #Construção da Matriz Omega
        Omega = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Omega[i, j] = y.iloc[i] * y.iloc[j] * K[i, j]
        
        #Construção da Matriz A
        H = Omega + (1/tau) * np.identity(n_samples)
        A = np.block([[np.array([0]), np.expand_dims(y, axis = 1).T],
                    [np.expand_dims(y, axis = 1), H]])

        #Construção do Vetor B
        B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                            np.expand_dims(np.ones(n_samples), axis = 1)), axis=0)
        
        #Resultados
        resultados = {'A': A,
                    "B": B}
        
        return(resultados)
    
    @staticmethod
    #Método para realizar a predição utilizando os multiplicadores de lagrange
    #ótimos estimados
    def predict_class(alphas, b, gamma, kernel, X_treino, y_treino, X_teste):
        #Inicialização
        alphas = np.array(alphas)
        estimado = np.zeros(X_teste.shape[0])
        n_samples_treino = X_treino.shape[0]
        n_samples_teste = X_teste.shape[0]
        K = np.zeros((n_samples_teste, n_samples_treino))
        
        #Construção da matriz de Kernel
        for i in range(n_samples_teste):
            for j in range(n_samples_treino):
                
                if kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X_teste.iloc[i, :], X_treino.iloc[j, :])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X_teste.iloc[i, :], X_treino.iloc[j, :])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X_teste.iloc[i, :], X_treino.iloc[j, :])
                
            #Realização da predição
            estimado[i] = np.sign(np.sum(np.multiply(np.multiply(alphas, y_treino), K[i])) + b)
        
        return estimado
    


