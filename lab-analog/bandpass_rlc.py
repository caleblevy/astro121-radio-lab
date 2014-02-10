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

def lowpass(frequency, resistor, capacitor):
    """
    Plot the gain of a low-pass filter.
    """
    resistor = float(resistor)
    capacitor = float(capacitor)

    square_term = 2 * np.pi * frequency * resistor * capacitor
    return 1.0 / np.sqrt(1.0 + square_term ** 2)

def highpass(frequency, resistor, capacitor):
    numerator = 2 * np.pi * frequency * resistor * capacitor
    return numerator * lowpass(frequency, resistor, capacitor)

def transfer(frequency, resistor, capacitor, inductor):
    """
    Transfer equation of the RLC circuit used in the lab.
    """
    resistor  = float(resistor)
    capacitor = float(capacitor)
    inductor  = float(inductor)

    numerator = 2 * np.pi * frequency * inductor
    denominator  = (1 - (2 * np.pi *frequency)**2 * inductor*capacitor) ** 2
    denominator *= resistor ** 2
    denominator += numerator ** 2
    return numerator / np.sqrt(denominator)

def rolloff(resistor, capacitor, inductor):
    """
    Returns a tuple contianing the two rolloff points of the filter.
    """
    resistor  = float(resistor)
    capacitor = float(capacitor)
    inductor  = float(inductor)

    # There are 4 main terms in the frequency equations
    term1 = 1.0 / (capacitor * inductor)
    term2 = 1.0 / (2.0 * (resistor*capacitor)**2)
    term3 = 1.0 / (resistor**2 * capacitor**3 * inductor)
    term4 = term2 ** 2

    # Combine the terms and return the frequencies
    sumfreq = term1 + term2
    innersqrt = np.sqrt(term3 + term4)
    freqs  = np.sqrt(np.array([sumfreq - innersqrt, sumfreq + innersqrt]))
    freqs /= 2 * np.pi * 1e6 # Units are MHz
    return tuple(freqs)

if __name__ == '__main__':
    # Circuit component values
    resistor = 33
    capacitor = 22000e-12
    inductor = 1e-6

    # Get the frequency and the gain
    frequency = np.logspace(2, 8, 10000)
    gain = 20 * np.log10(transfer(frequency, resistor, capacitor, inductor))

    # Get the resonance and rolloff points, units of MHz
    resonance = 1e-6 / np.sqrt(capacitor * inductor) / (2*np.pi)
    roll_low, roll_high = rolloff(resistor, capacitor, inductor)
    delta_f = 1e-6 / (2*np.pi * resistor * capacitor)
    gain_carrier = transfer(1.045e6, resistor, capacitor, inductor)

    # Display some results
    print 'Resonance frequency:', resonance, 'MHz'
    print 'Rolloff points:'
    print '\tLow:', roll_low, 'MHz'
    print '\tHigh:', roll_high, 'MHz'
    print '\t\Delta f_{exact}:', roll_high - roll_low, 'MHz'
    print '\t\Delta f_{Q}:', delta_f, 'MHz'
    print 'Gain at 1.045 MHz:', gain_carrier, '=',
    print 20*np.log10(gain_carrier), 'dB'

    # Plot the transfer function
    plt.plot(frequency, gain)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Bode plot of an RLC bandpass filter')
    plt.xscale('log')
    plt.tight_layout()

    # Plot the low pass filter gain
    resistor  = 150
    capacitor = 10e-9
    frequency = np.logspace(1, 6, 10000)
    gain = 20 * np.log10(lowpass(frequency, resistor, capacitor))
    plt.figure()
    plt.plot(frequency, gain)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Bode plot of an RC lowpass filter')
    plt.axis([np.min(frequency), np.max(frequency), np.min(gain), 1.0])
    plt.xscale('log')
    plt.tight_layout()

    plt.show()
