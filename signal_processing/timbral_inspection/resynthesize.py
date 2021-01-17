#Andrew Sivaprakasam
#Purdue University
#Email: asivapr@purdue.edu

#DESCRIPTION: Code written to isolate the magnitudes of harmonics of a
#given f_0 for a given audiofile/stimulus.

#Additional Dependencies: scipy, numpy, matplotlib
# pip3 install scipy
# pip3 install numpy
# pip3 install matplotlib

#May require ffmpeg on Ubuntu/Linux as well
# sudo apt-get install ffmpeg

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def extract_harmonics(fname, fs = 44100, f_0 = 440, n_harms = 3):
    fs, x  = wavfile.read(fname)
    #x = np.array(aud[0])
    t_vect = np.arange(0,len(x))/fs
    f_vect = np.arange(1,n_harms+1)*f_0;
    #plt.plot(t_vect,x)

    ## TODO: Try applying dpss to this. Might result in more accurate
    ## magnitudes?

    freq_time = np.multiply(np.asmatrix(f_vect).T,np.asmatrix(t_vect))
    x_sin = np.multiply(np.asmatrix(x),np.sin(2*np.pi*freq_time))
    x_cos = np.multiply(np.asmatrix(x),np.cos(2*np.pi*freq_time))
    sin_sum = np.sum(x_sin,1);
    cos_sum = np.sum(x_cos,1);

    mags = np.sqrt(np.multiply(sin_sum,sin_sum) + np.multiply(cos_sum,cos_sum))
    mags = np.squeeze(np.asarray(mags))/np.max(mags)
    #plt.stem(f_vect,mags)

    return [f_vect, mags, x, fs]

########################## IMPLEMENTATION #####################################

from signal_processing import pure_tone_complex, sound, magphase
import matplotlib.pyplot as plt

#Can use the below line in Atom when running Hydrogen
#%matplotlib inline

harmonics = 6;

extract = extract_harmonics('instruments/trombone_A4_normal.wav', fs = 44100, f_0 = 440, n_harms = harmonics);

fs_Hz = extract[3];
dur_sec = 2;
amp = extract[1];
phi = np.zeros(harmonics);
freq_Hz = extract[0];

tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi,'sq')
tone = tone[1]/np.max(tone[1]);
tone = tone*.75
plt.figure(2)
plt.plot(tone);
plt.xlim([0,4400])
sound(tone,fs_Hz,'resynth.wav',1)

## TODO: Clean up plots, try to directly compare DFT to extracted harmonics

plt.figure(0)
plt.stem(extract[0],extract[1])

plt.figure(1)
fig, (ax1,ax2) = magphase(extract[2],extract[3],x_axislim = [0,np.max(extract[0])])
