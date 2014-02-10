#!/usr/bin/env python2.7

################################################################################
## This script is for creating pdf frames of a traveling AM wave.
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
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt

def usage(code):
    """
    Display help options and exit.
    """
    print 'Usage:', os.path.basename(sys.argv[0]), '[options]'
    print 'Flags:'
    print '    -h: Print this help message and exit.'
    print '    -e: File extension to save the frames with (default: pdf).'
    print '    -n: Number of frames (required).'
    print '    -o: Output directory to save the frames to (required).'
    sys.exit(code)

def cwave(time, frequency, phase):
    """
    Carrier wave, assumed to have unity amplitude.
    """
    return np.cos(time * 2 * np.pi * frequency + phase)

def swave(time, frequency, phase, offset=1.0):
    """
    Signal wave, also has unity amplitude.
    """
    return offset + cwave(time, frequency, phase)

if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'he:n:o:')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)

    # Read options
    nframes = None
    outdir  = None
    extension = 'pdf'
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-e':
            extension = arg
        elif opt == '-n':
            nframes = int(arg)
        elif opt == '-o':
            outdir = os.path.abspath(arg)
    if nframes == None:
        print 'ERROR: No frame number specified.'
        usage(1)
    if outdir == None:
        print 'ERROR: No output directory specified.'
        usage(1)
    else:
        os.system('mkdir -p ' + outdir)

    # Carrier and signal frequencies.
    signal  = 440.0
    carrier = 10 * signal
    total_time = 2.0 / signal

    time = np.linspace(0, total_time, 1000)
    for i in range(nframes):
        phase = 2 * np.pi * i / nframes

        carrier_wave = cwave(time, carrier, phase)
        signal_wave  = swave(time, signal, phase, 1.25)

        plt.figure()
        plt.plot(time, carrier_wave * signal_wave, 'b')
        plt.plot(time, signal_wave, 'r')
        plt.axis('off')

        frame_name = outdir + '/am-wave-' + str(i) + '.' + extension
        print 'Saving frame:', frame_name
        plt.savefig(frame_name, bbox_inches='tight')
