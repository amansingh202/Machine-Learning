#!/usr/bin/env python
# coding: utf-8

# In[1079]:


'''
Name: Aman Kumar
Hawk ID: A20538809
Assignment 2
CS 584
'''


# In[1080]:


import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.utils import resample
import csv


# # Model to string method

# In[1081]:


def model_to_string(coeffs, m, n):
    ones = f"{coeffs[0]} "
    cosines = [f"{coeffs[j]:+} cos(2π * {j})" for j in range(1, m + 1)]
    sines = [f"{coeffs[j]:+} sin(2π * {j - m})" for j in range(m + 1, m + n + 1)]
    return ones + " ".join(cosines) + " " + " ".join(sines)


# # Fitting the model

# In[1082]:


#method to fit model
def fit_model(data, m, n):
    x_sample = data['x'].values.reshape(-1, 1)
    y_sample = data['y'].values.reshape(-1, 1)
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2 * np.pi * j * x_sample) for j in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * j * x_sample) for j in range(1, n + 1)])[:, :, 0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)

    u, s, vT = np.linalg.svd(dmatrix, full_matrices=False)
    uT = np.transpose(u)
    v = np.transpose(vT)
    s_inv = np.power(s, -1)
    p_inv = np.dot(v, np.dot(np.diag(s_inv), uT))
    coeffs = np.dot(p_inv, y_sample)
    model_stringified = model_to_string(coeffs.reshape(-1), m, n)

    outputs = np.dot(dmatrix, coeffs)
    resids = y_sample - outputs
    rmse = np.sqrt(np.mean(np.square(resids.reshape(-1))))

    return coeffs


# # Generating the data

# In[1083]:


#method to generate data
def generate_data(sample_size, noise_std, file_name):
    rng = np.random.default_rng()
    x_sample = rng.uniform(-10, 10, sample_size)
    noise = rng.normal(0, noise_std, sample_size)
    offset = rng.uniform(1)
    y_sample = signal.sawtooth(2 * np.pi * x_sample + offset) + noise
    
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in zip(x_sample, y_sample):
            writer.writerow([x, y])
    
    data = pd.DataFrame({'x': x_sample, 'y': y_sample})
    return data


# # Log likelihood method

# In[1084]:


#method to calculate log likelihood 
#m is the number of cosine terms
#n is the number of sine terms


def calculate_log_likelihood(data, coeffs, m, n):
    
    #extract the input features 
    x_in = data['x'].values.reshape(-1, 1)
    
    #extract the target variables
    y_in = data['y'].values.reshape(-1, 1)
    
    ones = np.ones_like(x_in)
    cosines = np.array([np.cos(2 * np.pi * i * x_in) for i in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * i * x_in) for i in range(1, n + 1)])[:, :, 0].T
    
    #datrix matrix of intercepts, sines and cosines
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)
    
    op = np.dot(dmatrix, coeffs)
    rem = y_in - op
    rss = np.sum(rem**2)
    
    n_samples = len(data)
    
    #number of parameters
    num_params = m + n + 1 
    deg = n_samples - num_params 
    
    #calculation of log likelihood
    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * rss / deg)  # assuming normal distribution
    
    return log_likelihood


# # AIC Calculation

# In[1085]:


#calculate AIC by using log likelihood


def calculate_aic(data, coeffs, m, n):
    #k is the number of parameters
    k = m + n +1
    
    #AIC is calculated using the formula 
    #AIC = 2*k - 2*log(L)
    #where L is the likelihood value 
    #this is calculated by calculating the likelihood function
    aic = 2*k - 2*np.log(calculate_log_likelihood(data, coeffs, m, n))
    return aic


# # RMSE's calculation

# In[1086]:


#calculating the root mean square error

def calculate_rmse(data, coeffs, m, n):
    
    #x is the input features
    x_sample = data['x'].values.reshape(-1, 1)
    
    #y is the target features
    y_sample = data['y'].values.reshape(-1, 1)
    
    #ones cosines and sines values
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2 * np.pi * i * x_sample) for i in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * i * x_sample) for i in range(1, n + 1)])[:, :, 0].T
    
    #dmatrix of sines, cosines and intercept
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)

    outputs = np.dot(dmatrix, coeffs)
    resids = y_sample - outputs
    
    #rmse is the sqrt of mean of (y_sample - outputs)^2
    rmse = np.sqrt(np.mean(np.square(resids.reshape(-1))))
    return rmse


# # K_fold_cross_validation

# In[1087]:


#method to calculate the k fold cross validation

def k_fold_cross_validation(data, m, n, k):
    
    #k number of splits
    kfc = KFold(n_splits=k)
    rmses = []
    
    #iterate over each fold and split the data into the training and test sets
    for train_i, test_i in kfc.split(data):
        train_data = data.iloc[train_i]
        test_data = data.iloc[test_i]
        
        #fits the regression model with the training data and computing the coefficients
        coeffs = fit_model(train_data, m, n)
        
        #calculates root mean square error on the testing data by calling the calculate_rmse function
        rmse = calculate_rmse(test_data, coeffs, m, n)
        rmses.append(rmse)
        
    #calculating the mean of the rmses values 
    return np.mean(rmses)


# # Bootstrap Resampling

# In[1088]:


#method to compute bootstrap resampling to evaluate the performance of the model
#num_bootstraps is the number of bootstrap samples to generate

def bootstrap(data, m, n, num_bs):
    rmses = []
    
    #iterate throught the num_bootstraps to generate multiple bootstrap samples
    for _ in range(num_bs):
        
        #bs_data is the bootstrap data
        bs_data = resample(data, replace=True)
        
        #compute coefficients as per the bootstrap data
        coeffs = fit_model(bs_data, m, n)
        
        #calculate rmse based on the data, coeffs and the parameter values
        rmse = calculate_rmse(data, coeffs, m, n)
        rmses.append(rmse)
        
    return np.mean(rmses)


# # Bias variance computation

# In[1089]:


#method to calculate the bias and variance of the model for different values of m and n

def calculate_bias_variance(data, coeffs, m, n):
    #extract the input features and the target variables from the data
    x_in = data['x'].values.reshape(-1, 1)
    y_in = data['y'].values.reshape(-1, 1)
    
    #intercepts, sines and cosines values for the dmatrix
    ones = np.ones_like(x_in)
    cos = np.array([np.cos(2 * np.pi * i * x_in) for i in range(1, m + 1)])[:, :, 0].T
    sine = np.array([np.sin(2 * np.pi * i * x_in) for i in range(1, n + 1)])[:, :, 0].T
    
    #matrix for the sine, cosine and intercept terms
    dmatrix = np.concatenate([ones, cos, sine], axis=1)
    
    # predicted value of target values
    y_predicted = np.dot(dmatrix, coeffs)
    
    #calculate bias from the sample and the mean of predicted values
    bias = np.mean((y_in - np.mean(y_predicted)) ** 2)
    
    #calculate variance from the difference of predicted and the mean of the predicted values
    variance = np.mean((y_predicted - np.mean(y_predicted)) ** 2)
    
    return bias, variance


# In[ ]:


def main():
    input_file = "sample.csv"

    data = pd.read_csv(input_file)
    
    #noise values from 0.01 to 0.1
    noise_stds = [0.01, 0.05, 0.1]
    
    #range of values of m 
    m_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    #n values 
    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    #variables to determine the value of m and n for min aic
    min_aic = np.inf
    best_aic_m = None
    best_aic_n = None
    
    #variables to determine the value of m and n for min values of k_5_fold rmse
    min_k_5_fold_rmse = np.inf
    best_k_5_fold_m = None
    best_k_5_fold_n = None
    best_k_5_fold = None
    
    #variables to determine the value of m and n for min values of k_10_fold rmse
    min_k_10_fold_rmse = np.inf
    best_k_10_fold_m = None
    best_k_10_fold_n = None
    best_k_10_fold = None
    
    #variables to determine the value of m and n for min values of bootstrap samples
    best_rmse = np.inf
    best_bs_m = None
    best_bs_n = None
    
    

    #for size in sample_sizes:
    
    #iterating through each value of noise passed
    for noise in noise_stds:
        #iterating through each value of parameter m
        for m in m_values:
            #iterarting through each value of parameter n
            for n in n_values:
                print(f"Noise Level: {noise}, m: {m}, n: {n}")
                
                #generate the data by calling the generate data function
                synthetic_data = generate_data(len(data), noise, 'sample.csv')
                
                #compute coefficients from the fit model function
                coeffs = fit_model(synthetic_data, m, n)
                model_stringified = model_to_string(coeffs.reshape(-1), m, n)
                #this will print the model coefficients
                print("Model:", model_stringified)
                
                #compute rmse by calling the calculate_rmse function
                rmse = calculate_rmse(synthetic_data, coeffs, m, n)
                print("RMSE:", rmse)
                
                #calculation of AIC 
                aic = calculate_aic(synthetic_data, coeffs, m, n)
                print("AIC:", aic)
                if aic < min_aic:
                    min_aic = aic
                    best_aic_m = m
                    best_aic_n = n
                
                #computation of k_fold_rmse for k = 5
                k_fold_rmse_5 = k_fold_cross_validation(synthetic_data, m, n, 5)
                print("5-Fold Cross Validation RMSE:", k_fold_rmse_5)
                
                #computation of k_fold_rmse for k = 10
                k_fold_rmse_10 = k_fold_cross_validation(synthetic_data, m, n, 10)
                print("10-Fold Cross Validation RMSE:", k_fold_rmse_10)
                
                #computing value of m and n for min value of k_fold rmse for k = 5
                if k_fold_rmse_5 < min_k_5_fold_rmse:
                    min_k_5_fold_rmse = k_fold_rmse_5
                    best_k_5_fold_m = m
                    best_k_5_fold_n = n
                
                #computing value of m and n for min value of k_fold rmse for k = 10
                if k_fold_rmse_10 < min_k_10_fold_rmse:
                    min_k_10_fold_rmse = k_fold_rmse_10
                    best_k_10_fold_m = m
                    best_k_10_fold_n = n
                
                #computation of bootstrap below
                #number of bootstrap samples that we are taking
                num_bootstraps = 100
                bootstrap_rmse = bootstrap(synthetic_data, m, n, num_bootstraps)
                print("Bootstrap RMSE:", bootstrap_rmse)
                
                #computing value of m and n for min value of bootstrap rmse
                if bootstrap_rmse < best_rmse:
                    best_rmse = bootstrap_rmse
                    best_bs_m = m
                    best_bs_n = n
                    
                
                #computing the bias and variance of the model
                bias, variance = calculate_bias_variance(synthetic_data, coeffs, m, n)
                print(f"Bias: {bias}")
                print(f"Variance: {variance}")
                    
    #values of m and n for each method      
    print(f"Minimum AIC: {min_aic}, m: {best_aic_m}, n: {best_aic_n}")
    print(f"Minimum k-5_Fold Cross Validation RMSE: {min_k_5_fold_rmse}, m: {best_k_5_fold_m}, n: {best_k_5_fold_n}")
    print(f"Minimum k-10_Fold Cross Validation RMSE: {min_k_10_fold_rmse}, m: {best_k_10_fold_m}, n: {best_k_10_fold_n}")
    print(f"Minimum Bootsrap value: {best_rmse}, m: {best_bs_m}, n: {best_bs_n}")
    print()



    

if __name__ == "__main__":
    main()


# In[ ]:




