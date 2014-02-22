#!/usr/bin/env python2.7

################################################################################
## This script is for using the ROACH to sample signals.
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
import subprocess as spr

# Usage function is defined before the checking if one is on the ugastro network
# so that it can be displayed if one fails to meet that condition.
def usage(code):
    """
    Display help options and exit.
    """
    print 'Usage:', os.path.basename(sys.argv[0]), '[options]'
    print 'Flags:'
    print '    -h: Print this help message and exit.'
    print '    -f: Signal frequency.'
    print '    -F: Sample frequency.'
    print '    -o: Output filename base (required).'
    print '    -s: SRS function generator number.'
    sys.exit(code)

# Verify that this script is being run on the ugastro network.
if 'ugastro.berkeley.edu' not in socket.gethostname():
    print 'ERROR: You are not on the ugastro.berkeley.edu network.'
    usage(1)

# The radiolab module can now be imported, since it is only available on the
# ugastro.berkeley.edu network.
import radiolab as rl

def collect_raw_data():
    """
    This function collects data from the FPGA. No calibration is done
    on the data in this function.
    """
    # A bof process should be running, but if it isn't, start one.
    bof_pid = get_bof_pid()
    if bof_pid == None:
        the_wolf_is_loose()
        bof_pid = get_bof_pid()

    # Trigger commands
    trig_0 = 'ssh root@roach \'echo -n -e ' + r'"\x00\x00\x00\x00"'
    trig_1 = 'ssh root@roach \'echo -n -e ' + r'"\x00\x00\x00\x01"'
    ioreg_path = '/proc/' + bof_pid + '/hw/ioreg'

    print 'Collecting data...'
    os.system(trig_1 + ' > ' + ioreg_path + '/trig\'')
    time.sleep(1)
    os.system(trig_0 + ' > ' + ioreg_path + '/trig\'')
    time.sleep(1)

    # Generate commands to gram data from the ROACH
    grab_data = ['ssh', 'root@roach', 'cat']
    bram_names = ['/adc_bram', '/cos_bram', '/sin_bram', '/mix_bram']
    data_cmds = [grab_data + [ioreg_path + bram] for bram in bram_names]

    # Grab the data from the ROACH
    raw_data = []
    for cmd in data_cmds:
        subpr = spr.Popen(cmd, stdout=spr.PIPE)
        data, _ = subpr.communicate()
        raw_data.append(np.array(data))

    return raw_data

def get_bof_pid():
    """
    Determine the existence of a BOF process on the ROACH. The PID of
    the process is returned as a string.
    """
    # Detect
    find_pid = ['ssh', 'root@roach', 'ps', 'aux']
    subpr = spr.Popen(find_pid, stdout=spr.PIPE)
    processes, _ = subpr.communicate()
    bof_process = filter(lambda s: 'bof' in s, processes.split('\n'))

    # Because of the limitations of the ROACH hardware, only one bof process
    # can be running at a time, so bof_process[0] will return the process string
    # of the process every single time it exists.
    if len(bof_process):
        return bof_process[0].split()[1]

def make_total_destroy():
    """
    This function searches for a .bof process running on the ROACH and
    then proceeds to killing it if it exists.
    """
    # Detect BOF process running.
    bof_pid = get_bof_pid()

    # Determine whether or not to kill the BOF process
    if bof_pid == None:
        print 'WARNING: No BOF process running on the ROACH.'
    else:
        bof_pid = bof_pid
        print 'WARNING: Killing BOF process', bof_pid + '.'
        kill_cmd = 'ssh root@roach kill ' + bof_pid
        os.system(kill_cmd)

def the_wolf_is_loose():
    """
    Starts the BOF process used to collect data.
    """
    # I don't want to assume that the BOF process name is invariant. The newest
    # file containing the proper pattern will be the BOF file to be used.
    find_bof = ['ssh', 'root@roach', 'ls', '-t', '/boffiles']
    subpr = spr.Popen(find_bof, stdout=spr.PIPE)
    bof_files, _ = subpr.communicate()
    bof_process = filter(lambda s: 'adc_snaps' in s, bof_files.split('\n'))[0]
    bof_process = '/boffiles/' + bof_process

    # Start the BOF process
    print 'Starting BOF process...'
    os.system('ssh root@roach ' + bof_process + ' &')
    print 'PID of BOF process:', get_bof_pid()

if __name__ == '__main__':
    # Parse options from the command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:F:lm:o:')
    except getopt.GetoptError as err:
        print 'ERROR:', str(err)
        usage(1)

    # Read options
    outfbase = None
    loc_osc = False
    sample_rate = 200e6
    signal_freq = 1e6
    mix_freq    = None
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-f':
            signal_freq = float(arg)
        elif opt == '-F':
            sample_rate = float(arg)
        elif opt == '-l':
            loc_osc = True
        elif opt == '-m':
            mix_freq = float(arg)
        elif opt == '-o':
            outfbase = os.path.abspath(arg)
            outdir = os.path.dirname(outfbase)
    if outfbase == None:
        print 'ERROR: No output filename specified.'
        usage(1)
    else:
        os.system('mkdir -pv ' + outdir)

    # Initialize the generator
    print 'Setting up the function generator.'
    rl.set_srs(1, vpp=0, off=0, pha=0)
    rl.set_srs(2, vpp=0, off=0, pha=0)
    time.sleep(1)
    rl.set_srs(1, dbm=-10, freq=signal_freq)
    if mix_freq != None and not loc_osc:
        rl.set_srs(2, dbm=-10, freq=mix_freq)
    time.sleep(1)

    # Kill existing bof processes and start a new one.
    make_total_destroy()
    the_wolf_is_loose()

    metadata = np.array([ ('SAMPRATE', sample_rate)
                        , ('SIGFREQ', signal_freq)
                        , ('MIXFREQ', mix_freq)
                        , ('FPGA LO', loc_osc)
                        ], dtype=object)

    # Mix with the FPGA local oscillator.
    # TODO actually implement this...
    if mix_freq != None and loc_osc:
        pass

    # Collect the raw data.
    raw_data = collect_raw_data()

    # Save the raw data.
    raw_filename = outfbase + '-raw.npz'
    print 'Saving raw data to', raw_filename + '.'
    np.savez(raw_filename, metadata, *raw_data)

    # Convert the data to usable numbers ad save it.
    numeric_filename = outfbase + '-numeric.npz'
    print 'Saving converted data to', numeric_filename + '.'
    numeric_data = [np.frombuffer(data, '>i4') for data in raw_data]
    np.savez(numeric_filename, metadata, *numeric_data)

    # Kill the BOF process once all of the data is collected.
    make_total_destroy()
