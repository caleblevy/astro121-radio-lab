#!/usr/bin/env python2.7

################################################################################
## This file computes the DFT of signals measured with single_ch.py.
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
import numpy.fft as fft
import scipy.integrate as spi
import matplotlib as mpl

def usage(code):
    """
    Display help options and exit.
    """
    print 'Usage:', os.path.basename(sys.argv[0]), '[options]'
    print 'Flags:'
    print '    -h: Print this help message and exit.'
    print '    -i: Input filename (required).'
    print '    -n: Number of cycles of the input wave to plot.'
    print '    -N: Number of samples to use.'
    print '    -o: Output filename base.'
    print '    -p: Percentage of the signal to use.'
    print '    -q: Do not display plots.'
    sys.exit(code)

def dft_sum(signal):
    """
    Compute the Discrete Fourier Transform. This code is designed to be
    brute-force and probably shouldn't be used except as a proof of
    concept. Just use a pre-packaged FFT.
    """
    # Damn it feels good to be a gangster...
    indices = np.arange(len(signal))
    sine_waves = np.exp(-2j * np.pi * indices / len(signal))
    return np.array(map(lambda k: np.sum(signal * sine_waves**k), indices))

def dft_freq(freq, time, signal):
    """
    Compute the DFT of a signal at a specific frequency. This function
    is recursive, and is probably going to be very slow for large
    signals. This is only really going to be used to test the frequency
    resolution of the DFT, since this directly computes the Fourier
    integral, whereas the definition of the DFT only involves the
    number of samples used in a signal.
    """
    if hasattr(freq, '__iter__'):
        dft = np.array(map(lambda f: dft_freq(f, time, signal), freq))
    else:
        dft = spi.trapz(signal * np.exp(-2j*np.pi*freq*time), time)

    return dft

def get_ps(dft, signal):
    """
    Compute the power spectrum using FFT functions that both take in
    and return one numpy array. This is just a convienent one-liner.
    """
    return np.abs(dft(signal)) ** 2

def plot_units(ticks, tick_unit):
    """
    Convert the unit of a plot into something more readable:
    Example: 0.0001 s -> 0.1 ms or 100 microseconds.

    Returns: (tick_unit, ticks)
    """
    units = {0:r'', 1:r'k', 2:r'M', 3:r'G', -1:r'm', -2:r'$\mu ', -3:'n'}
    ticks = np.array(ticks[:])

    # Determine the scale of the ticks
    rms_tick = np.sqrt(np.mean(ticks**2))
    unit_index = 0
    if rms_tick > 1:
        while rms_tick > 1:
            rms_tick /= 1e3
            unit_index += 1
        unit_index -= 1
    else:
        while rms_tick < 1:
            rms_tick *= 1e3
            unit_index -= 1

    ticks *= 10.0 ** (-3*unit_index)

    # Generate the unit label.
    tick_unit = units[unit_index] + tick_unit
    if unit_index == -2 or '$' in tick_unit:
        tick_unit += '$'

    return (tick_unit, ticks)

def savefig(figname):
    """
    Print a message to tell that a figure is being saved, then save the
    figure to both PDF and PNG formats. Since the figure is being saved
    to multiple formats, the extension should be excluded from the
    figure name when passing it to this function.
    """
    print 'Saving plot to', figname + '.pdf.'
    print 'Saving plot to', figname + '.png.'
    plt.savefig(figname + '.pdf', bbox_inches='tight')
    plt.savefig(figname + '.png', bbox_inches='tight')

def switch_halves(array):
    """
    For switching the front and back halves of an array.
    """
    arrlen = len(array)
    pivot = (arrlen + arrlen % 2) / 2
    return np.r_[array[pivot:], array[:pivot]]

if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:n:N:o:p:q')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)

    # Read options
    infilename = None
    ncycles    = 6
    nsamples   = None
    outdir     = None
    outfbase   = None
    ptsratio   = None
    disp_plots = True
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-i':
            infilename = os.path.abspath(arg)
        elif opt == '-n':
            ncycles = int(arg)
        elif opt == '-N':
            nsamples = int(arg)
        elif opt == '-o':
            outfbase = os.path.abspath(arg)
            outdir = os.path.dirname(outfbase)
        elif opt == '-p':
            if arg[-1] == '%':
                ptsratio = float(arg[:-1]) / 100.0
            else:
                ptsratio = float(arg)
            if ptsratio > 1:
                print 'ERROR: Percentage of signal too high!'
                usage(1)
        elif opt == '-q':
            mpl.use('Agg')
            disp_plots = False
    if infilename == None:
        print 'ERROR: No input file.'
        usage(1)
    if outdir != None:
        os.system('mkdir -pv ' + outdir)

    # matplotlib.use must be called before import matplotlib.pyplot
    import matplotlib.pyplot as plt

    # Extract the signal from the .npz file
    print 'Importing data from', infilename + '.'
    infile = np.load(infilename)
    file_info = dict(infile['arr_0'])
    sample_rate = file_info['SAMPRATE']
    signal_freq = file_info['SIGFREQ']
    volt_peak   = file_info['VOLTPP']

    # Determine the number of samples to use
    if nsamples == None and ptsratio == None:
        nsamples = 2 * (infile['arr_1'].shape[0] / 2)
    elif nsamples == None and ptsratio != None:
        nsamples = 2 * (int(ptsratio * infile['arr_1'].shape[0]) / 2)
    elif nsamples != None and ptsratio == None:
        nsamples = min(nsamples, len(infile['arr_1']))
    elif nsamples != None and ptsratio != None:
        print 'ERROR: Cannot use -N and -p simultaneously.'
        usage(1)

    # Compute t and signal(t)
    time = np.arange(nsamples, dtype=float) / sample_rate
    signal = infile['arr_1'][:nsamples]
    infile.close()

    # Compute the Fourier transform of the signal.
    print 'Computing the DFT of the signal.'
    pspec_dft = switch_halves(get_ps(dft_sum, signal))
    pspec_np  = switch_halves(get_ps(fft.fft, signal))
    freq = switch_halves(fft.fftfreq(nsamples, 1.0 / sample_rate))

    # Get the fourier transform for a very fine frequency space
    print 'Computing the Fourier Transform over a fine frequency resolution.'
    freq_res_th = sample_rate / nsamples
    fine_freq = np.linspace(10 * freq_res_th, 20 * freq_res_th, 10001)
    fine_dft  = np.abs(dft_freq(fine_freq, time, signal)) ** 2

    # Attempt to estimate the resolution of the DFT.
    # TODO get the uncertainty.
    print 'Estimating the uncertainty of the Fourier transform.'
    time_spec  = get_ps(fft.fft, fine_dft / np.max(fine_dft))
    time_dt   = fft.fftfreq(len(time_spec), fine_freq[1] - fine_freq[0])
    freq_res = 1.0 / np.abs(time_dt[np.argmax(time_spec[1:len(time_spec)/2])+1])

    # Generate a line displaying $\Delta f$. The lower bound of the line will
    # start at the secondmost minimum peak frequency in the DFT. The peaks are
    # to be found by making partitions of the DFT based on the frequency
    # resolution that was measured.
    parts = [0] # Partitions of the dft
    # Approximate number of indices for the partitions
    index_df  = 1000
    max_inds  = []
    while parts[-1] + index_df < len(fine_freq):
        parts.append(parts[-1] + index_df)
        max_inds.append(parts[-2] + np.argmax(fine_dft[parts[-2]:parts[-1]]))
    max_freqs = map(lambda i: fine_dft[i], max_inds)
    min_ind = np.argmin(max_freqs)
    max_inds.remove(max_inds[min_ind])
    max_freqs.remove(max_freqs[min_ind])
    max_ps = max(max_freqs)

    start_freq   = fine_freq[max_inds[np.argmin(max_freqs)]]
    end_freq     = start_freq + freq_res

    freq_height  = min(max_freqs)
    freq_height1 = min(freq_height + max_ps / 10, 3.5 * max_ps / 4.5)
    freq_height2 = freq_height1 + max_ps / 20


    print 'Measured frequency resolution:', freq_res, 'Hz'
    print 'Predicted frequency resolution:', freq_res_th, 'Hz'

    # Simulate what the sampler should measure.
    print 'Simulating what the sampler should measure.'

    # Phase offset to give to the wave, based on the data.
    phase = np.pi / 2 - 2 * np.pi * signal_freq * time[np.argmax(signal)]

    # Prediction of what should be measured.
    signal_sim = volt_peak * np.sin(2 * np.pi * signal_freq *time + phase)

    # Number of points to plot is determined by the number of cycles of the
    # input wave that are getting plotted.
    nplot = min(nsamples, ncycles * int(sample_rate / signal_freq))

    # Generate properly sampled signal to show what the SRS was inputting.
    fine_step = 1 / (100.0 * signal_freq)
    fine_time = np.arange(0, time[nplot], fine_step)
    fine_signal = volt_peak * np.sin(2*np.pi*signal_freq*fine_time + phase)

    # Time to generate some plots...
    print 'Plotting...'

    # Plot of the time domain of the signal, plus the power spectrum.
    plt.figure(figsize=(14,5.25))
    plt.subplot(1,2,1)
    plt.plot(fine_time, fine_signal, 'k--', label='Input Signal')
    plt.plot(time[:nplot], signal[:nplot], 'b', label='Observed Signal')
    ax = plt.gca()
    time_u, time_ticks = plot_units(ax.get_xticks(), 's')
    ax.set_xticklabels(map(str, time_ticks))
    plt.xlabel(r'Time (' + time_u + ')')
    plt.ylabel(r'Signal (V)')
    plt.title('Time Domain')
    plt.legend(loc='upper left', prop={'size':12})

    plt.subplot(1,2,2)
    plt.plot(freq, pspec_np)
    ax = plt.gca()
    freq_u, freq_ticks = plot_units(ax.get_xticks(), 'Hz')
    ax.set_xticklabels(map(str, freq_ticks))
    plt.xlabel(r'Frequency (' + freq_u + ')')
    plt.ylabel('Power Spectrum')
    plt.title('Frequency Domain')

    plt.yscale('log')
    plt.tight_layout()
    if outfbase != None:
        savefig(outfbase + '-domains')

    # Plot an overlay of numpy's FFT vs my DFT
    plt.figure(figsize=(14,5.25))
    plt.subplot(1,2,1)
    plt.plot(freq, pspec_dft,      label='Homemade DFT')
    # Multiplying by 100 gives a visible offset when plotting on a log scale.
    plt.plot(freq, 100 * pspec_np, label='Numpy\'s FFT $\\times$ 100')
    ax = plt.gca()
    freq_u, freq_ticks = plot_units(ax.get_xticks(), 'Hz')
    ax.set_xticklabels(map(str, freq_ticks))
    plt.xlabel(r'Frequency (' + freq_u + ')')
    plt.ylabel('Power Spectrum')
    plt.yscale('log')
    plt.legend(loc='lower center', prop={'size':12})

    plt.subplot(1,2,2)
    plt.plot(freq, (pspec_dft - pspec_np) / pspec_np)
    ax = plt.gca()
    freq_u, freq_ticks = plot_units(ax.get_xticks(), 'Hz')
    ax.set_xticklabels(map(str, freq_ticks))
    plt.xlabel(r'Frequency (' + freq_u + ')')
    plt.title('Relative difference of DFT computations')
    plt.tight_layout()
    if outfbase != None:
        savefig(outfbase + '-compare-dft')

    # Plot the fine frequency space.
    plt.figure(figsize=(7,5.25))
    plt.plot(fine_freq, fine_dft)
    ax = plt.gca()
    freq_u, freq_ticks = plot_units(ax.get_xticks(), 'Hz')
    ax.set_xticklabels(map(str, freq_ticks))
    plt.xlabel(r'Frequency (' + freq_u + ')')
    plt.ylabel('Power Spectrum')
    plt.title('Testing the frequency resolution of the DFT')
    res_label = 'Predicted $\Delta f =' + str(round(freq_res_th, 1)) + '$ Hz'
    res_label += '\nMeasured $\Delta f = ' + str(round(freq_res, 1)) + '$ Hz'
    pred_label='Predicted $\Delta f = ' + str(round(freq_res_th, 1)) + '$ Hz'
    meas_label='Measured $\Delta f = ' + str(round(freq_res, 1)) + '$ Hz'
    plt.plot([start_freq, end_freq],
             [freq_height2, freq_height2],
             '#555753',
             label=pred_label,
             linestyle='--',
             linewidth=3,
             marker='d',
             markeredgewidth=1,
             markersize=6)
    plt.plot([start_freq, end_freq],
             [freq_height1, freq_height1],
             'k',
             label=meas_label,
             linewidth=3,
             marker='d',
             markeredgewidth=1,
             markersize=6)
    plt.legend(loc='upper left', prop={'size':12})
    plt.tight_layout()
    if outfbase != None:
        savefig(outfbase + '-freq-res')

    # Plot the simulated signal
    plt.figure(figsize=(7,5.25))
    ax = plt.gca()
    plt.plot(fine_time, fine_signal, 'k', label='Input Signal')
    plt.plot(time[:nplot], signal_sim[:nplot], 'b--')
    plt.plot(time[:nplot], signal_sim[:nplot], 'bo', label='Predicted signal')
    time_u, time_ticks = plot_units(ax.get_xticks(), 's')
    ax.set_xticklabels(map(str, time_ticks))
    plt.xlabel(r'Time (' + time_u + ')')
    plt.ylabel(r'Signal (V)')
    plt.title('Simulation of the sampler')
    plt.legend(loc='upper left', prop={'size':12})
    if outfbase != None:
        savefig(outfbase + '-sim')

    if disp_plots:
        plt.show()
