#!/usr/bin/env python2.7

################################################################################
## This script is for recording a mixed signal with the pulsar machine.
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

# Usage function is defined before testing if one is on the ugastro network.
# This scipt will only work on that network, since it interfaces to hardware in
# the lab.
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
    print '    -o: Output filename base (required).'
    print '    -s: SRS function generator number.'
    print '    -v: Voltage level of the input signal.'
    print '    -V: Voltage level of the mixing signal.'
    sys.exit(code)

# Verify that this script is being run on the ugastro network.
if 'ugastro.berkeley.edu' not in socket.gethostname():
    print 'ERROR: You are not on the ugastro.berkeley.edu network.'
    usage(1)

# The radiolab module to control the function generator can now be imported.
import radiolab as rl

if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:F:in:o:v:V:')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)

    # Read options
    store_int = False
    outdir = None
    volt_sig = 1.0
    volt_mix = 1.0
    sample_rate = 20e6
    signal_freq = 1e6
    mix_freq = 1e6
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-f':
            signal_freq = float(arg)
        elif opt == '-F':
            sample_rate = float(arg)
        elif opt == '-i':
            store_int = True
        elif opt == '-o':
            filename = os.path.abspath(arg)
            outdir = os.path.dirname(filename)
        elif opt == '-v':
            volt_sig = float(arg)
        elif opt == '-V':
            volt_mix = float(arg)
    if outdir == None:
        print 'ERROR: No output location specified.'
        usage(1)
    else:
        os.system('mkdir -pv ' + outdir)

    # Initialize the generator
    print 'Initializing the function generators.'
    rl.set_srs(1, dbm=0, off=0, pha=0)
    rl.set_srs(2, dbm=0, off=0, pha=0)
    time.sleep(1)
    print 'Setting SRS 1 to output at', signal_freq/1e6, 'MHz.'
    print 'Setting SRS 2 to output at', mix_freq/1e6, 'MHz.'
    rl.set_srs(1, vpp=volt_sig, freq=signal_freq)
    rl.set_srs(2, vpp=volt_mix, freq=mix_freq)
    time.sleep(1)

    # Don't sample for longer than 1/10th of a second.
    sample_time = 0.1
    nsamples = min(262143, int(sample_rate * sample_time))

    # Get the samples.
    samples = rl.sampler(nsamples, sample_rate, dual=False, integer=store_int)

    # Useful header containing information about the data being collected.
    metadata = np.array([ ('SAMPRATE', sample_rate)
                        , ('SIGFREQ', signal_freq)
                        , ('MIXFREQ', mix_freq)
                        , ('VOLTSIG', volt_sig)
                        , ('VOLTMIX', volt_mix)
                        ], dtype=object)

    # Save an npz file containing the data and a header
    print 'Recording', nsamples, 'samples to', filename + '.'
    np.savez(filename, metadata, samples)
