#!/usr/bin/env python2.7

################################################################################
## This script plots the transfer function of an RLC bandpass.
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
import matplotlib.pyplot as plt

def transfer(frequency, resistor, capacitor, inductor):
    """
    Transfer equation of the RLC circuit used in the lab.
    """
    resistor   = float(resistor)
    capacitor  = float(capacitor)
    inductor   = float(inductor)

    numerator = 2 * np.pi * frequency * inductor
    denominator  = (1 - (2 * np.pi *frequency)**2 * inductor*capacitor) ** 2
    denominator *= resistor ** 2
    denominator += numerator ** 2
    return numerator / np.sqrt(denominator)

if __name__ == '__main__':
    frequency = np.logspace(4, 8, 10000)
    resistor = 33
    capacitor = 22000e-12
    inductor = 1e-6
    gain = transfer(frequency, resistor, capacitor, inductor)

    resonance = 1e-6 / np.sqrt(capacitor * inductor) / (2*np.pi)
    print 'Resonance frequency:', resonance, 'MHz'

    plt.plot(frequency, 20 * np.log10(gain))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Bode plot of an RLC bandpass filter')
    plt.xscale('log')

    plt.show()
