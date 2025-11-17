import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler 

#variables 
SEED = 1161
N_TRAIN = 9
N_VALID = 100
N_TEST  = 100
Ms = np.arange(1,N_TRAIN+1) #1<M<N
i = np.arange(-14, 2+1) #log10lambda = i. 10^-14 < i < 10^2

#Equation for f_opt
def f_opt(x):
    return np.sin(2.0 * np.pi * x)

#Generate the test, vaildation, and training set 
def generate_sets(seed, N_train, N_valid, N_test):
    x_tr = np.linspace(0, 1, N_train)
    x_va = np.linspace(0, 1, N_valid)
    x_te =np.linspace(0, 1, N_test)
    np.random.randn(seed)
    #var is 0.2 the noise is random 
    t_tr = f_opt(x_tr) + 0.2*np.random.randn(N_train)
    t_va = f_opt(x_va) + 0.2*np.random.randn(N_valid)
    t_te = f_opt(x_te) + 0.2*np.random.randn(N_test)
    return x_tr, t_tr, x_va, t_va, x_te, t_te

#design a matrix of a size 
def design_matrix(x, M):
    return np.vstack([x**m for m in range(M + 1)]).T

#create a matrix of coeff
def fit_least_squares(x, t):
    A=np.dot(x.T, x)
    A1=np.linalg.inv(A)
    b=np.dot(x.T, t)
    w=np.dot(A1,b)
    return w

#calculate the RMSE 
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def standardize (XX_train, XX_valid):
    # Ensure 2D. Convert inputs to NumPy arrays because the sklearn only takes 2D
    XX_train = np.asarray(XX_train)
    XX_valid = np.asarray(XX_valid)
    #If someone passed a 1-D vector reshape to 2-D (N, 1)
    if XX_train.ndim == 1:
        XX_train = XX_train.reshape(-1, 1)
    if XX_valid.ndim == 1:
        XX_valid = XX_valid.reshape(-1, 1)
        
    #will store the mean μ_j and std σ_j for each feature 
    sc = StandardScaler()
    #Fit on the training data
    XX_train_std = sc.fit_transform(XX_train)
    XX_valid_std = sc.transform(XX_valid)
    return XX_train_std, XX_valid_std

def L2reg (X_st, lam, t):
    # Build B  with size P×P
    N, P = X_st.shape
    B = np.zeros((P, P))
    B[1:, 1:] = 2.0 * lam * np.eye(P - 1)   # add the diag as 2*lambda
    # L2 Regularization 
    # w = ((Xst^T Xst) + (N/2) * B)^(-1) Xst^T t
    A  = np.dot(X_st.T,X_st) # (P, P)
    A1 = A + (N / 2.0) * B # (P, P)
    b  = np.dot(X_st.T, t) # (P,)  this will be t_tr
    w_L2 = np.linalg.solve(A1, b) # (P,)
    return w_L2

def main():
    x_tr, t_tr, x_va, t_va, x_te, t_te = generate_sets(SEED, N_TRAIN, N_VALID, N_TEST)
    #init for getting error plor
    rmse_train_list = []
    rmse_valid_list = []
    #go through 1<M<N
    for M in Ms:
        # Design matrices
        w_x_tr = design_matrix(x_tr, M)
        w_x_va = design_matrix(x_va, M)
        
        #get the coeff matrix 
        w = fit_least_squares(w_x_tr, t_tr)
        
        # Predictions
        y_tr = np.dot(w_x_tr, w)
        y_va = np.dot(w_x_va, w)
    
        # RMSE
        r_tr = rmse(t_tr, y_tr)
        r_va = rmse(t_va, y_va)
        rmse_train_list.append(r_tr)
        rmse_valid_list.append(r_va)
    
        #plot each M with all the test training data and predicted curve with this M
        plt.title(f'Polynomial M={M} ')
        x_plot = np.linspace(0.0, 1.0, 400)
        f_opt_plot = np.sin(2 * np.pi * x_plot)
        plt.plot(x_plot, f_opt_plot, color='black', label="Optimal Prediction")
        plt.scatter(x_tr, t_tr, color='magenta', label= "Traning Data")
        plt.plot(x_tr, y_tr, color='green', label = "Prediction Curve") #y_one is predicted values
        plt.scatter(x_te, t_te, color='blue', label = "Test Data") #test data 
        plt.legend()
        plt.show() 

    #PLot errors
    plt.title('Training RMSE vs Validation RMSE')
    plt.plot(Ms, rmse_train_list, color='magenta', label = "Train RMSE")
    plt.plot(Ms, rmse_valid_list, color ='green', label = "Validation RMSE")
    plt.legend()
    plt.show()
    
    #do lambdas for M=9 in bishops the 3 lambdas are ln(lam) = 10^i)
    #penalty term is to offset overfitting so checking on M=9
    rmse_valid_list = []
    rmse_train_list = []

    Z_tr = design_matrix(x_tr, M)[:, 1:] # shape (N, M)
    Z_va = design_matrix(x_va, M)[:, 1:] # shape (N_val, M)
    # Standardize features on training 
    Z_tr_std, Z_va_std = standardize(Z_tr, Z_va)  # returns standardized (N, M) and scaler
    
    #Add bias column of 1  ,X_st has shape (N, P=M+1)
    X_st = np.hstack([np.ones((Z_tr_std.shape[0], 1)), Z_tr_std]) # (N, 10)
    X_st_va = np.hstack([np.ones((Z_va_std.shape[0], 1)), Z_va_std]) # (N_val, 10)
    for lamb in i:
        #L2 Regularization 
        lam = np.power(10, lamb, dtype=float)#10^i
        w_L2 = L2reg(X_st, lam,t_tr) #get w from L2
        y_tr_L2 = np.dot(X_st, w_L2) #get y predicted curve from w 
        y_va_L2 = np.dot(X_st_va, w_L2)
        r_tr = rmse(t_tr, y_tr_L2)#get the error
        r_va = rmse(t_va, y_va_L2)
        rmse_train_list.append(r_tr)
        rmse_valid_list.append(r_va)

        #plot
        plt.title(f'Polynomial M={M} with L2 Regularization (λ={lam})')
        x_plot = np.linspace(0.0, 1.0, 400)
        f_opt_plot = np.sin(2 * np.pi * x_plot)
        plt.plot(x_plot, f_opt_plot, color='black', label="Optimal Prediction")
        plt.scatter(x_tr, t_tr, color='magenta', label="Training Data")
        plt.plot(x_tr, y_tr_L2, color='green', label = "Prediction Curve") #y_one is predicted values
        plt.scatter(x_te, t_te, color='blue', label = "Test Data") #test data 
        plt.legend()
        plt.show()

    
    #plot the RMSE against log lambda
    plt.plot(i, rmse_train_list, label='Train RMSE')
    plt.plot(i, rmse_valid_list, label='Validation RMSE')
    plt.xlabel(r'$\log_{10}\lambda$')
    plt.ylabel('RMSE')
    plt.title('Train vs Validation RMSE across $\log_{10}\lambda$')
    plt.legend()
    plt.show()
    
main()