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
    #Funções auxiliares para a proposta FSLM_LSSVM improved
    #---------------------------------------------------------------
    @staticmethod
    def purity_level(X, y):
        '''
        Função para determinar o nível de pureza de uma determinado conjunto com
        base nas alterações de sinal do rótulo de cada amostra
        INPUT:
            X - Array de features (Array de dimensão n x p);
            y - Array de targets (Array de dimensão n x 1);
            index - Conjunto de índices correspondentes a um subconjunto de X.
            
        OUTPUT:
            pureza - Nível de pureza dada pelas trocas de sinal no rótulo de cada
            amostra do subconjunto em análise.
        '''
        #Incialização
        contador = 0
        y_anterior = y[0]
        
        for i in range(len(y)):

            if  y[i] * y_anterior < 0:
                contador += 1
            
            y_anterior = y[i]
        
        return(contador)
    
    @staticmethod
    def cluster_optimum(X, y, eps):
        '''
        Método para determinação do cluster com maior nível de impureza, onde o 
        processo de clusterização é baseado no algoritmo DBSCAN.
        INPUT:
            X - Array de features (Array de dimensão n x p);
            y - Array de targets (Array de dimensão n x 1);
            eps - máxima distância entre duas amostras para uma ser considerada 
            vizinhança da outra (float, default = 0.5).
            
        OUTPUT:
            índices do cluster maior impureza.
        '''
        
        #Convertendo dataframe
        X = pd.DataFrame(X)
        y = pd.Series(y, name = "y")
        
        #Clusterização utilizando DBSCAN
        clustering = DBSCAN(eps = eps).fit(X)
        
        #Recuperando os índices
        cluster = pd.Series(clustering.labels_, name = "cluster")
        df = pd.concat([cluster, X, y], axis = 1)
        purity = df.groupby('cluster').apply(Utils.purity_level, df.y)
        
        return(df.where(df.cluster == purity.idxmax()).dropna(axis = 0).index)
    
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta Fast Nesterov LSSVM_ADMM 
    #---------------------------------------------------------------
    @staticmethod
    def Sthresh(x, gamma):
        return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)
    
    @staticmethod
    def ADMM(A, y):

        m, n = A.shape
        w, v = np.linalg.eig(A.T.dot(A))
        MAX_ITER = 10000

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        xhat = np.zeros([n, 1])
        zhat = np.zeros([n, 1])
        u = np.zeros([n, 1])

        "Calcular os coeficientes e tamanho do passo"
        l = sqrt(2*log(n, 10))
        rho = 1/(np.amax(np.absolute(w)))

        "Pre-ca´cula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < MAX_ITER):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - u))

            "z minimização via soft-thresholding"
            zhat = Utils.Sthresh(xhat + u, l/rho)

            "atualização dos multiplicadores"
            u = u + xhat - zhat

            i = i+1
        return zhat, rho, l
    
    @staticmethod
    def Fast_ADMM(A, y, l, rho):

        m, n = A.shape
        w, v = np.linalg.eig(A.T.dot(A))
        MAX_ITER = 100000

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        #Inicialização
        xhat = np.zeros([n, 1])
        
        z = np.zeros([n, 1])
        zhat = np.zeros([n, 1])

        u = np.zeros([n, 1])
        uhat = np.zeros([n, 1])

        alpha = 1 #termo de momento

        "Calcular os coeficientes e tamanho do passo"
        #l = sqrt(2*log(n, 10))
        #rho = 1/(np.amax(np.absolute(w)))

        "Pre-cálcula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < MAX_ITER):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - uhat))

            "z minimização via soft-thresholding"
            z_anterior = z
            z = Utils.Sthresh(xhat + uhat, l/rho)

            "atualização dos multiplicadores"
            u_anterior = u
            u = uhat + xhat - z

            "atualização do termo de momento"
            alpha_anterior = alpha
            alpha = (1+sqrt(1+4*(alpha**2)))/2

            "Atualização do vetor z"
            zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

            "Atualização dos multiplicadores de Lagrange"
            uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

            i = i+1
        return zhat, rho, l

    @staticmethod
    def Fast_ADMM_restart(A, y, tau, neta):

        m, n = A.shape
        w, v = np.linalg.eig(A.T.dot(A))
        MAX_ITER = 50000

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        #Inicialização
        xhat = np.zeros([n, 1])
        
        z = np.zeros([n, 1])
        zhat = np.zeros([n, 1])

        u = np.zeros([n, 1])
        uhat = np.zeros([n, 1])

        alpha = 1 #termo de momento
        c = 0

        "Calcular os coeficientes e tamanho do passo"
        l = sqrt(2*log(n, 10))
        rho = 1/(np.amax(np.absolute(w)))

        "Pre-cálcula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < MAX_ITER):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - uhat))

            "z minimização via soft-thresholding"
            z_anterior = z
            z = Utils.Sthresh(xhat + uhat, l/rho)

            "atualização dos multiplicadores"
            u_anterior = u
            u = uhat + xhat - z

            #Regra para reinicialização
            c_anterior = c
            c = (1/tau)*(np.linalg.norm(u - uhat)**2) + tau*(np.linalg.norm(z - zhat)**2)

            alpha_anterior = alpha
            if c < neta*c_anterior:
                "atualização do termo de momento"
                alpha = (1+sqrt(1+(4*alpha**2)))/2

                "Atualização do vetor z"
                zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

                "Atualização dos multiplicadores de Lagrange"
                uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

            else:
                alpha = 1
                zhat = z_anterior
                uhat = u_anterior
                c = c/neta
            i = i+1
        return zhat, rho, l
    
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
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta RFSLM_LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def Construct_A_b_RFSLM(X, y, kernel, tau, gamma):
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
        
        #Construção da Matriz Omega
        Omega = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Omega[i, j] = y[i] * y[j] * K[i, j]
        
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
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta LARS_LM_LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def lars(A, v, ni, epsilon = 0.1):

        '''
        Interface do Método
        Ação: Método para solucionar um sistema da forma Au = v
        utilizando o algoritmo LARS
        
        INPUT:
        A: Matriz de coeficientes do sistema de dimensão (N+1) x (N+1) (Dataframe);
        v: Vetor de termos independentes do sistema de dimensão (N+1) x 1 (Series);
        ni: Parâmetro de regularização de Tikhonov (escalar);
        epsilon: Tolerância (escalar)

        OUTPUT:
        u: solução aproximada do sistema de dimensão (N+1) x 1 (Series).
        '''
        
        #Convertendo para pamadas Dataframe
        A = pd.DataFrame(A)
        v = pd.DataFrame(v)
        
        #Normalização das colunas da matriz dos coeficientes A e
        #do vetor de termos independentes v
        A_centered = A - A.mean(axis = 0)
        A_centered /= np.linalg.norm(A_centered, axis = 0)
        v_centered = v - v.mean()

        A = np.array(A_centered)
        v = np.array(v_centered).squeeze()

        #Dimensões da matriz dos coeficientes
        n_samples, n_features = A.shape
        mu = np.zeros_like(v)
        beta = np.zeros(n_samples)
        erros = []

        for i in range(n_features):
            
            #Calculando a correlação com o resíduo e
            #determinado o conjunto ativo
            c = A @ (v - mu)
            c_abs = np.abs(c)
            c_max = c_abs.max()

            active = np.isclose(c_abs, c_max)
            signs = np.where(c[active] > 0, 1, -1)

            A_active = signs * A[:, active]

            #Calculando a atualização de Levenberg-Marquardt
            G = A_active.T @ A_active + ni * np.diag(A_active.T @ A_active)
            Ginv = np.linalg.inv(G)

            aux = Ginv.sum() ** (-0.5)
            w = aux * Ginv.sum(axis = 1)
            delta = A_active @ w

            gamma = c_max/aux

            if not np.all(active):
                a = A.T @ delta
                complement = np.invert(active)
                cc = c[complement]
                ac = a[complement]
                candidates = np.concatenate([(c_max - cc) / (aux - ac),
                                            (c_max + cc) / (aux + ac)])
                gamma = candidates[candidates >= 0].min()
            
            
            #Determinação do erro
            erros.append((np.linalg.norm(v - mu))**2)

            #Atualizações
            mu += gamma * delta
            beta[active] += gamma * signs

        #Resultados
        resultados = {"mult_lagrange": beta[1:],
                      "b": beta[0],
                      "Erros": erros,
                      "Indices_multiplicadores": active,
                      "Estimativa": mu}

        #Retornando os multiplicadores de Lagrange finais
        return(resultados)


