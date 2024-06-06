#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import csv
import numpy
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.utils import resample

def model_to_string(coeffs, m, n):
    ones = f"{coeffs[0]} "
    cosines = [f"{coeffs[j]:+} cos(2\u03C0 * {j})" for j in range(1, m+1) ]
    sines = [f"{coeffs[j]:+} sin(2\u03C0 * {j-m})" for j in range(m+1, m+n+1)]
    return  ones + " ".join(cosines) + " " + " ".join(sines)

def fit_model(m, n, x_sample, y_sample,  input_file="sample.csv"):
    data = []
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append(line)
    ones = numpy.ones((len(x_sample), 1))
    cosines = numpy.array([numpy.cos(2*numpy.pi*j*x_sample) for j in range(1,2*m+1)]).T  
    sines = numpy.array([numpy.sin(2*numpy.pi*j*x_sample) for j in range(1,2*n+1)]).T 
    dmatrix = numpy.concatenate([ones, cosines, sines], axis=1) 
    u, s, vT = numpy.linalg.svd(dmatrix, full_matrices=False)
    uT = numpy.transpose(u)
    v = numpy.transpose(vT)
    s_inv = numpy.power(s, -1)
    p_inv = numpy.dot(v, numpy.dot(numpy.diag(s_inv), uT))
    coeffs = numpy.dot(p_inv, y_sample)
    model_stringified = model_to_string(coeffs.reshape(-1), m, n)
    outputs = numpy.dot(dmatrix, coeffs)
    resids = y_sample - outputs
    rmse = numpy.sqrt(numpy.mean(numpy.square(resids.reshape(-1))))
    return rmse, model_stringified, coeffs, dmatrix

def generate_data(size=10000, noise=0.1, output_file="sample.csv"):
    rng = numpy.random.default_rng()
    x_sample = rng.uniform(-10,10, size)
    noise = rng.normal(0, noise, size)
    offset = rng.uniform(1)
    y_sample = signal.sawtooth(2*numpy.pi*x_sample + offset) + noise
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in zip(x_sample, y_sample):
            writer.writerow([x,y])  
    return x_sample, y_sample

def calculate_aic(n, mse, num_params):
    aic = n * numpy.log(mse) + 2 * num_params
    return aic

def k_fold_cv(m, n, x_sample, y_sample, k):
    kf = KFold(n_splits=k)
    scores = []
    for train_index, test_index in kf.split(x_sample):
        x_train, x_test = x_sample[train_index], x_sample[test_index]
        y_train, y_test = y_sample[train_index], y_sample[test_index]
        rmse, _, _, _ = fit_model(m, n, x_train, y_train)  
        scores.append(rmse)
    return numpy.mean(scores)

def calculate_bias(y_true, y_pred):
    return numpy.mean(y_true - y_pred)

def calculate_variance(y_pred):
    return numpy.var(y_pred)

def bootstrap_evaluation(m, n, x_sample, y_sample):
    x_bs_sampled, y_bs_sampled = resample(x_sample, y_sample)
    rmse, _, _, _ = fit_model(m, n, x_bs_sampled, y_bs_sampled) 
    return rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", help="Number of cosine terms", type=int)
    parser.add_argument("n", help="Number of sine terms", type=int)
    parser.add_argument("-s", "--size", help="Sample size", type=int, default=10000)
    parser.add_argument("-n", "--noise", help="Magnitude of the noise", type=float, default=0.1)
    parser.add_argument("-f", "--output_file", help="Path and name of output file", default="sample.csv")
    args = parser.parse_args()
    x_sample, y_sample = generate_data(args.size, args.noise)
    rmse, coeffs, dmatrix = fit_model(args.m, args.n, x_sample, y_sample)
    print(f"RMSE: {rmse}")
    aic = calculate_aic(len(y_sample), rmse**2 , args.m + args.n + 1)  
    print(f"AIC: {aic}")
    avg_score_5_fold = k_fold_cv(args.m, args.n, x_sample, y_sample, k=5)  
    avg_score_10_fold = k_fold_cv(args.m, args.n, x_sample, y_sample, k=10)  
    print(f"Average Score with 5-Fold CV: {avg_score_5_fold}")
    print(f"Average Score with 10-Fold CV: {avg_score_10_fold}")
    y_pred = numpy.dot(dmatrix, coeffs)
    bias = calculate_bias(y_sample, y_pred)
    variance = calculate_variance(y_pred)
    print(f"Bias: {bias}")
    print(f"Variance: {variance}")
    bootstrap_score = bootstrap_evaluation(args.m, args.n, x_sample, y_sample)  
    print(f"Bootstrap Score: {bootstrap_score}")

if __name__=="main_":
    main()


# In[ ]:




