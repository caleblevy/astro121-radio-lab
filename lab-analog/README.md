Analog lab programs
===============================

bandpass\_rlc.py
----------------------

This script plots the transfer function of the RLC bandpass filter that was
built, using the nominal values of the circuit components used. It also
creates a plot of the gain function of the RLC bandpass multiplied with the
low-pass filter in the envelope detector.

central\_limit.py
---------------------

This script does a demonstration of the Central Limit Theorem. What it does is
create a list of sampled means, and creates a histogram from that least. Each
sampled mean is generated from an array of random samples from some parent
distribution, where the size of the array is the sample size. The script has
options to set the number of bins in the histogram it produces, the number of
samples used to compute each mean, the number of means to sample, and which
random number distribution to draw from.

envelope.py
-------------------

This script creates frames for an AM wave animation. This is just for embedding
a cute animation of an AM wave in a LaTeX document. This script has options to
set the file extension used for the frames, the number of frames, and the output
directory to save the frames to. It should be noted that, unfortunately, only
Adobe Reader will display animations in a PDF document. However, output of this
script can be saved in PNG format and imagemagick can be used to create a GIF
image out of the frames.  
  
Creating a GIF image:  
There are two steps to creating a GIF image of a sample AM signal. The first is
to generate the image frames:  
$ python envelope.py -e png -n \<numframes\> -o /path/to/frame/dir/  
Next, one must go to the directory with the frames for the gif and use
imagemagick to combine them into a GIF image:  
$ cd /path/to/frame/dir/  
$ convert -delay \<time\> -loop 0 am-wave-\* name\_of\_gif\_image.gif

root\_n.py
-------------------

This scipt is used to demonstrate the 1 / \sqrt{N} dependence of the standard
deviation of a sampled mean. This script takes in no options from the user.
