#!/usr/bin/env python2.7

################################################################################
## This script is for testing the central limit theorem.
## Copyright (C) 2014  Isaac Domagalski: idomagalski@berkeley.edu
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

import os
import sys
import getopt
import numpy as np
import scipy.stats as sps
import numpy.random as npr
import matplotlib.pyplot as plt

def usage(code):
    """
    Display help options and exit.
    """
    print 'Usage:', os.path.basename(sys.argv[0]), '[options]'
    print 'Flags:'
    print '    -h: Print this help message and exit.'
    print
    print 'Sampling and binning options:'
    print '    -b: Number of bins in the histogram (default: 10)'
    print '    -n: Number of samples from the distribution (default: 10).'
    print '    -N: Number of sampled means (default: 1000).'
    print
    print 'When no random number distribution is set, numbers are selected'
    print 'from an even distribution ranging from 0 to 1.'
    print 'Random number distributions:'
    print '    -g: Normal distribution.'
    print '    -G: Gamma distribution.'
    print '    -l: Lognormal distribution.'
    print '    -p: Poisson distribution.'
    print '    -r: Rayleigh distribution.'
    sys.exit(code)

def get_rand(rand_func, nsamples, *args):
    """
    Gets random samples from some numpy.random distribution.
    """
    if len(args):
        samples = rand_func(*args, size=nsamples)
    else:
        samples = rand_func(size=nsamples)
    return samples

def gaussian(x, magnitude, mean=0, sigma=1):
    magnitude *= sigma * np.sqrt(2*np.pi)
    return magnitude * sps.norm.pdf(x, loc=mean, scale=sigma)

# Run everything.
if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hb:n:N:gGlpr')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)
        usage(1)
    if len(args):
        print 'ERROR: Bad input.'
        usage(1)

    # Read options
    nbins    = 10
    niter    = 1000
    nsamples = 10
    rand_name = 'uniform distribution'
    rand_func = npr.random
    rand_args = []
    rand_mean = 0.5
    rand_std  = 1.0 / np.sqrt(12)
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        # Select sampling and plotting options
        elif opt == '-b':
            nbins = int(arg)
        elif opt == '-n':
            nsamples = int(arg)
        elif opt == '-N':
            niter = int(arg)
        # Select random distributions
        elif opt == '-g':
            rand_name = 'Gaussian distribution'
            rand_func = npr.normal
            rand_args = [0.0, 1.0]
            rand_mean, rand_std = rand_args
        elif opt == '-G':
            rand_name = 'Gamma distribution'
            rand_func = npr.gamma
            rand_args = [1.0, 1.0]
            rand_mean = np.product(rand_args)
            rand_std  = np.sqrt(rand_args[0] * rand_args[1]**2)
        elif opt == '-l':
            rand_name = 'Lognormal distribution'
            rand_func = npr.lognormal
            rand_args = [0.0, 1.0]
            rand_mean = np.exp(rand_args[0] + rand_args[1]**2 / 2)
            rand_std  = np.exp(2*rand_args[0] + rand_args[1]**2)
            rand_std *= np.exp(rand_args[1]**2) - 1.0
            rand_std  = np.sqrt(rand_std)
        elif opt == '-p':
            rand_name = 'Poisson distribution'
            rand_func = npr.poisson
            rand_args = [10.0]
            rand_mean = rand_args[0]
            rand_std  = np.sqrt(rand_args[0])
        elif opt == '-r':
            rand_name = 'Rayleigh distribution'
            rand_func = npr.rayleigh
            rand_args = [1.0]
            rand_mean = rand_args[0] * np.sqrt(np.pi/2)
            rand_std  = rand_args[0] * np.sqrt(2 - np.pi/2)

    # The spread of the sampled means is equal to the spread of the parent
    # distribution divided by the root of the number of samples used to compute
    # the mean.
    rand_std /= np.sqrt(nsamples)

    # Get the sampled mean.
    mean_samples = []
    for i in range(niter):
        mean_samples.append(np.mean(get_rand(rand_func, nsamples, *rand_args)))

    # Plot a histogram of the sampled means
    n, bins, patches = plt.hist(mean_samples, nbins)

    # Plot a gaussian alongside the histogram.
    plt.plot(bins, gaussian(bins, max(n), rand_mean, rand_std), linewidth=3)

    # Format the plot
    plt.xlabel('Sampled mean')
    plt.ylabel('Counts')
    plt.title('Sampling from a ' + rand_name)
    plt.tight_layout()

    plt.show()
