

import argparse
import csv

import numpy
from scipy import signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", help="Sample size", type=int, default=10000)
    parser.add_argument("-n", "--noise", help="Magnitude of the noise", type=float, default=0.1)
    parser.add_argument("-f", "--output_file", help="Path and name of output file", default="sample.csv")
    args = parser.parse_args()

    rng = numpy.random.default_rng()
    x_sample = rng.uniform(-10,10, args.size)
    noise = rng.normal(0, args.noise, args.size)
    offset = rng.uniform(1)
    y_sample = signal.sawtooth(2*numpy.pi*x_sample + offset) + noise
    
    with open(args.output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in zip(x_sample, y_sample):
            writer.writerow([x,y])



if __name__=="__main__":
    main()

