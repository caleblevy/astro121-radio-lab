Digital lab programs
----------------------------

get\_spec\_1ch.py
------------------------

This script takes in the data that gets produced in single\_ch.py and produces a
spectrum, demonstrates a homemade DFT to numpy's FFT, and estimates the
frequency resolution of the FFT. 

    Usage: get_spec_1ch.py -i infile [options]  
    Flags:  
        -h: Print this help message and exit.  
        -i: Input filename (required).  
        -N: Number of samples to use.  
        -o: Output filename base.  
        -p: Percentage of the signal to use.  

single\_ch.py
-----------------------------

This script records data from a SRS function generator in the UC Berkeley
undergrad astronomy lab. This script checks that the computer running it is on
the ugastro.berkeley.edu network and if it is, then the script collects data.
The data is saved as an npz file.

    Usage: -o /out/filename/base [options]  
    Flags:  
        -h: Print this help message and exit.  
        -f: Signal frequency.  
        -F: Sample frequency.  
        -i: Store data as integers.  
        -n: Record several frequencies.    
                Frequencies: [step -> step * nfreq] * f_sample  
        -o: Output filename base (required).  
        -s: SRS function generator number.  
        -v: Peak-to-peak voltage level.  
    --------------------------------------------------------------------------------
    NPZ file contents:  
    'arr\_0': Information of the run that got collected (e.g. sample rate, etc...)  
    'arr\_1': The signal, measured in volts.
