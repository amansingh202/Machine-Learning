
import argparse
import csv

import numpy

def model_to_string(coeffs, m, n):
    ones = f"{coeffs[0]} "
    cosines = [f"{coeffs[j]:+} cos(2\u03C0 * {j})" for j in range(1, m+1) ]
    sines = [f"{coeffs[j]:+} sin(2\u03C0 * {j-m})" for j in range(m+1, m+n+1)]
    return  ones + " ".join(cosines) + " " + " ".join(sines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", help="Number of cosine terms", type=int)
    parser.add_argument("n", help="Number of sine terms", type=int)
    parser.add_argument("-f", "--input_file", help="Name of input data file", default="sample.csv")
    args = parser.parse_args()

    data = []
    with open(args.input_file, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append(line)


    x_sample = numpy.array([float(d["x"]) for d in data])
    y_sample = numpy.array([float(d["y"]) for d in data])
    x_sample = x_sample.reshape(-1, 1)
    y_sample = y_sample.reshape(-1, 1)
    ones = numpy.ones_like(x_sample)
    cosines = numpy.array([numpy.cos(2*numpy.pi*j*x_sample) for j in range(1,args.m+1)])[:,:,0].T
    sines = numpy.array([numpy.sin(2*numpy.pi*j*x_sample) for j in range(1,args.n+1)])[:,:,0].T
    dmatrix = numpy.concatenate([ones, cosines, sines], axis=1)
    
    u, s, vT = numpy.linalg.svd(dmatrix, full_matrices=False)
    uT = numpy.transpose(u)
    v = numpy.transpose(vT)
    s_inv = numpy.power(s, -1)
    p_inv = numpy.dot(v, numpy.dot(numpy.diag(s_inv), uT))
    coeffs = numpy.dot(p_inv, y_sample)
    model_stringified = model_to_string(coeffs.reshape(-1), args.m, args.n)
    print(model_stringified)
    outputs = numpy.dot(dmatrix, coeffs)
    resids = y_sample - outputs
    rmse = numpy.sqrt(numpy.mean(numpy.square(resids.reshape(-1))))
    print(f"RMSE: {rmse}")


if __name__=="__main__":
    main()
        
