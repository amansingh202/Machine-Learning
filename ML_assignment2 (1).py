#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.utils import resample

def model_to_string(coeffs, m, n):
    ones = f"{coeffs[0]} "
    cosines = [f"{coeffs[j]:+} cos(2π * {j})" for j in range(1, m + 1)]
    sines = [f"{coeffs[j]:+} sin(2π * {j - m})" for j in range(m + 1, m + n + 1)]
    return ones + " ".join(cosines) + " " + " ".join(sines)






# In[111]:


def fit_model(data, m, n):
    x_sample = data['x'].values.reshape(-1, 1)
    y_sample = data['y'].values.reshape(-1, 1)
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2 * np.pi * j * x_sample) for j in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * j * x_sample) for j in range(1, n + 1)])[:, :, 0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)

    coeffs = np.linalg.lstsq(dmatrix, y_sample, rcond=None)[0]
    return coeffs


# In[112]:


def generate_synthetic_data(sample_size, noise_std):
    x = np.linspace(0, 10, sample_size)
    sawtooth_wave = signal.sawtooth(2 * np.pi * x)
    noise = np.random.normal(0, noise_std, sample_size)
    y = sawtooth_wave + noise
    data = pd.DataFrame({'x': x, 'y': y})
    return data


# In[113]:


# def calculate_aic(num_samples, rmse, m, n):
#     k = m + n + 1
#     aic = num_samples * np.log(rmse) + 2*k
#     aic /= num_samples
#     return aic

#     log_likelihood = -0.5 * num_samples * np.log(2 * np.pi * (rmse ** 2) / num_samples)
    
#     aic = 2*k - 2*np.log(log_likelihood)
    
#     return aic
    


# In[114]:


def calculate_log_likelihood(data, coeffs, m, n):
    x_sample = data['x'].values.reshape(-1, 1)
    y_sample = data['y'].values.reshape(-1, 1)
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2 * np.pi * j * x_sample) for j in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * j * x_sample) for j in range(1, n + 1)])[:, :, 0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)
    
    outputs = np.dot(dmatrix, coeffs)
    residuals = y_sample - outputs
    rss = np.sum(residuals**2)
    
    n_samples = len(data)
    num_params = m + n + 1  # Number of parameters including intercept
    dof = n_samples - num_params  # Degrees of freedom
    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * rss / dof)  # assuming normal distribution
    
    return log_likelihood


# In[115]:


def calculate_aic(data, coeffs, m, n):
    k = m + n +1
    aic = 2*k - 2*np.log(calculate_log_likelihood(data, coeffs, m, n))
    return aic


# In[116]:


def calculate_rmse(data, coeffs, m, n):
    x_sample = data['x'].values.reshape(-1, 1)
    y_sample = data['y'].values.reshape(-1, 1)
    ones = np.ones_like(x_sample)
    cosines = np.array([np.cos(2 * np.pi * j * x_sample) for j in range(1, m + 1)])[:, :, 0].T
    sines = np.array([np.sin(2 * np.pi * j * x_sample) for j in range(1, n + 1)])[:, :, 0].T
    dmatrix = np.concatenate([ones, cosines, sines], axis=1)

    outputs = np.dot(dmatrix, coeffs)
    resids = y_sample - outputs
    rmse = np.sqrt(np.mean(np.square(resids.reshape(-1))))
    return rmse


# In[117]:


def k_fold_cross_validation(data, m, n, k):
    kf = KFold(n_splits=k)
    rmses = []
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        coeffs = fit_model(train_data, m, n)
        rmse = calculate_rmse(test_data, coeffs, m, n)
        rmses.append(rmse)
    return np.mean(rmses)


# In[118]:


def bootstrap(data, m, n, num_bootstraps):
    rmses = []
    for _ in range(num_bootstraps):
        boot_data = resample(data, replace=True)
        coeffs = fit_model(boot_data, m, n)
        rmse = calculate_rmse(data, coeffs, m, n)
        rmses.append(rmse)
    return np.mean(rmses)


# In[119]:


def main():
    input_file = "sample.csv"

    data = pd.read_csv(input_file)
    
    sample_sizes = [100, 500, 1000]
    noise_stds = [0.01, 0.05, 0.1]
    m_values = [3, 4, 5]
    n_values = [3, 4, 5]

    for size in sample_sizes:
        for noise in noise_stds:
            for m in m_values:
                for n in n_values:
                    print(f"Sample Size: {size}, Noise Level: {noise}, m: {m}, n: {n}")
                    synthetic_data = generate_synthetic_data(size, noise)
                    coeffs = fit_model(synthetic_data, m, n)
                    model_stringified = model_to_string(coeffs.reshape(-1), m, n)
                    print("Model:", model_stringified)
                    rmse = calculate_rmse(synthetic_data, coeffs, m, n)
                    print("RMSE:", rmse)
                    aic = calculate_aic(synthetic_data, coeffs, m, n)
                    print("AIC:", aic)
                    k_fold_rmse_5 = k_fold_cross_validation(synthetic_data, m, n, 5)
                    print("5-Fold Cross Validation RMSE:", k_fold_rmse_5)
                    k_fold_rmse_10 = k_fold_cross_validation(synthetic_data, m, n, 10)
                    print("10-Fold Cross Validation RMSE:", k_fold_rmse_10)
                    num_bootstraps = 100
                    bootstrap_rmse = bootstrap(synthetic_data, m, n, num_bootstraps)
                    print("Bootstrap RMSE:", bootstrap_rmse)
                    print()
            

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




