'''
Implementação da classe para a construção das funções de kernel
Data:15/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from numpy import linalg

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class Kernel:

    #Método Construtor
    def __init__(self, gamma, C = 1, d = 3, kernel = "gaussiano"):
        self.gamma = gamma
        self.C = C
        self.d = d
        self.kernel = kernel
    
    #Métodos Estáticos para as funções de kernel
    @staticmethod
    def linear_kernel(x, x_k):
        return np.dot(x, x_k)
    
    @staticmethod
    def polinomial_kernel(self, x, y):
        return (np.dot(x, y) + self.C)**self.d
    
    @staticmethod
    def gaussiano_kernel(gamma, x, y):
        return np.exp(-gamma * linalg.norm(x - y)**2)
    
    @staticmethod
    def Construct_A_B(X, y, kernel, gamma, tau):
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
                    K[i, j] = Kernel.linear_kernel(X[i], X[j])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X[i], X[j])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X[i], X[j])
        
        #--------------------------------------------------------------------------
        #Decomposição da matriz de kernel
        #--------------------------------------------------------------------------
        #Cholesky Incompleta
        P = np.linalg.cholesky(K + 0.01 * np.diag(np.full(K.shape[0], 1)))
        
        #Construção da matriz dos coeficiente A
        A = P.T
        
        #Construção do vetor de coeficientes b
        B = np.dot(linalg.inv(tau * np.identity(n_samples) + np.dot(P.T, P)), np.dot(P.T, y))
        B = np.expand_dims(B, axis = 1)
        
        #Resultados
        resultados = {'A': A,
                    "B": B}
        
        return(resultados)
    
    @staticmethod
    def Construct_A_B_regression(X, y, kernel, gamma, tau):
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
        n_samples, n_features = X.shape
    
        #Matriz de Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                
                #Kernel trick
                if kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X[i], X[j])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X[i], X[j])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X[i], X[j])
        
        #Construção da Matriz Omega
        Omega = K
    
        #--------------------------------------------------------------------------
        #Construção do sistema linear com matriz dos coeficientes
        #simétrica, definda positiva: Ax = B
        #--------------------------------------------------------------------------
        #Construção da matriz A
        H = Omega + (1/tau) * np.identity(n_samples)
        um_coluna = np.ones((n_samples))
        s = np.dot(um_coluna, np.linalg.inv(H).dot(um_coluna))
        zero_linha = np.zeros((1, n_samples))
        zero_coluna = np.zeros((n_samples, 1))
        A = np.block([[s, zero_linha], [zero_coluna, H]])
        
        #Construção do vetor B
        d1 = 0
        d2 = np.expand_dims(y, axis = 1)
        b1 = np.expand_dims(np.array(np.dot(um_coluna, np.linalg.inv(H).dot(y))), axis = 0)
        B = np.concatenate((np.expand_dims(b1, axis = 1), d2), axis = 0)
        B = np.squeeze(B)

        #Resultados
        resultados = {'A': A,
                      "B": B}
        
        return(resultados)