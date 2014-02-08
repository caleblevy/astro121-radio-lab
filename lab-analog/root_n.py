#!/usr/bin/env python2.7

################################################################################
## This script is for showing the 1/sqrt(N) dependence of the error of a mean.
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

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sample_sizes = np.array(range(10,101))
    sampled_stddev = []

    # Get the standard deviations of sampled means taken by sampling a normal
    # distribution. The sample size runs over a range of values. The percent
    # error on each standard deviation is computed from the number of means used
    # to compute the standard deviation as 1 / sqrt(2*nmeans - 2)
    nmeans = 100
    for n in sample_sizes:
        means = [np.mean(npr.normal(size=n)) for i in range(nmeans)]
        sampled_stddev.append(np.std(means, ddof=1))
    sampled_stddev = np.array(sampled_stddev)
    err_stddev = sampled_stddev / np.sqrt(2.0 * nmeans - 2.0)

    # Create labels for a plot, plot the sampled sizes and the standard
    # deviation of the means, compare it to 1 / sqrt(N), plot the errors.
    data_label = '$\sigma_{\langle x \\rangle}$'
    line_label = '$1 / \sqrt{N}$'
    plots = []
    plots += plt.plot(sample_sizes, sampled_stddev, 'bo')
    plots += plt.plot(sample_sizes, 1 / np.sqrt(sample_sizes), 'k', linewidth=2)
    plt.errorbar(sample_sizes,
                 sampled_stddev,
                 yerr=err_stddev,
                 fmt=None,
                 ecolor='k')
    plt.xlabel('Sample size')
    plt.ylabel(data_label)
    plt.title('Spread of sampled means')
    plt.legend(plots, (data_label, line_label))
    plt.show()
