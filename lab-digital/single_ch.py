#!/usr/bin/env python2.7

################################################################################
## This script is for recording signals from the SRS function generator.
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
import time
import getopt
import socket
import numpy as np

# Usage function is defined before the importing DFEC so that it can be used if
# the condition required for importing DFEC (being on the ugastro network) is
# not met.
def usage(code):
    """
    Display help options and exit.
    """
    print 'Usage:', os.path.basename(sys.argv[0]), '[options]'
    print 'Flags:'
    print '    -h: Print this help message and exit.'
    print '    -f: Signal frequency.'
    print '    -F: Sample frequency.'
    print '    -i: Store data as integers.'
    print '    -n: Record several frequencies.'
    print '            Frequencies: [step -> step * nfreq] * f_sample'
    print '    -o: Output filename base (required).'
    print '    -s: SRS function generator number.'
    print '    -v: Peak-to-peak voltage level.'
    sys.exit(code)

# Verify that this script is being run on the ugastro network.
if 'ugastro.berkeley.edu' not in socket.gethostname():
    print 'ERROR: You are not on the ugastro.berkeley.edu network.'
    usage(1)

# The DFEC module to control the function generator can now be imported.
import DFEC

if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:F:in:o:s:v:')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)

    # Read options
    store_int = False
    outdir = None
    srs_num = 1
    v_pp = 1.0
    nfreqs = None
    sample_rate = 20e6
    signal_freq = 10e6
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-f':
            signal_freq = float(arg)
        elif opt == '-F':
            sample_rate = float(arg)
        elif opt == '-i':
            store_int = True
        elif opt == '-n':
            nfreqs = int(arg)
        elif opt == '-o':
            outfbase = os.path.abspath(arg)
            outdir = os.path.dirname(outfbase)
        elif opt == '-s':
            srs_num = int(arg)
        elif opt == '-v':
            v_pp = float(arg)
    if outdir == None:
        print 'ERROR: No output location specified.'
        usage(1)
    else:
        os.system('mkdir -pv ' + outdir)

    # Initialize the generator
    DFEC.set_srs(srs_num, dbm=0, off=0, pha=0)
    DFEC.set_srs(srs_num, vpp=v_pp)

    # Just use 20 MHz for the sampling rate to start off. This means that
    # dual-mode must be off.
    sample_time = 0.1 # Don't sample for longer than 1/10th of a second
    nsamples = min(262143, int(sample_rate * sample_time))
    dual_mode = False

    # Get samples.
    if nfreqs != None:
        step = 0.1
        sig_freqs = np.linspace(step, nfreqs * step, nfreqs) * sample_rate

        for i in range(nfreqs):
            print 'Setting SRS', srs_num,
            print 'to output at', sig_freqs[i]/1e6, 'MHz.'
            DFEC.set_srs(srs_num, freq=sig_freqs[i])
            time.sleep(5) # The generator has some delay.

            # record voltage samples.
            print 'Sampling...'
            samples = DFEC.sampler(nsamples,
                                   sample_rate,
                                   dual=dual_mode,
                                   integer=store_int)

            # Save an npz file containing the data and a header
            filename = outfbase + '-' + str(i+1) + '.npz'
            print 'Recording', nsamples, 'samples to', filename + '.'
            metadata = np.array([ ('SAMPRATE', sample_rate)
                                , ('SIGFREQ', sig_freqs[i])
                                , ('VOLTPP', v_pp)
                                ], dtype=object)
            np.savez(filename, metadata, samples)

            # Organize formatting for printing output.
            if i + 1 < nfreqs:
                print
    else:
        print 'Setting SRS', srs_num,
        print 'to output at', signal_freq/1e6, 'MHz.'
        DFEC.set_srs(srs_num, freq=signal_freq)
        time.sleep(5) # The generator has some delay.

        # record voltage samples.
        print 'Sampling...'
        samples = DFEC.sampler(nsamples,
                               sample_rate,
                               dual=dual_mode,
                               integer=store_int)

        # Save an npz file containing the data and a header
        filename = outfbase + '.npz'
        print 'Recording', nsamples, 'samples to', filename + '.'
        metadata = np.array([ ('SAMPRATE', sample_rate)
                            , ('SIGFREQ', signal_freq)
                            , ('VOLTPP', v_pp)
                            ])
        np.savez(filename, metadata, samples)
