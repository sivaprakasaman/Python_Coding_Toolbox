#Andrew Sivaprakasam
#Purdue University
#Email: asivapr@purdue.edu

#DESCRIPTION: Code written to isolate the magnitudes of harmonics of a
#given f_0 for a given audiofile/stimulus.

#Additional Dependencies: librosa, numpy, matplotlib
# pip3 install librosa
# pip3 install numpy
# pip3 install matplotlib

#May require ffmpeg on Ubuntu/Linux as well
# sudo apt-get install ffmpeg

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb

def extract_harmonics(fname, fs = 44100, f_0 = 440, n_harms = 3):
    aud  = lb.load(fname,fs)
    x = np.array(aud[0])
    t_vect = np.arange(0,len(x))/fs
    f_vect = np.arange(1,n_harms+1)*f_0;
    #plt.plot(t_vect,x)

    ## TODO: Try applying dpss to this. Might result in more accurate 
    ## magnitudes.

    freq_time = np.multiply(np.asmatrix(f_vect).T,np.asmatrix(t_vect))
    x_sin = np.multiply(np.asmatrix(x),np.sin(2*np.pi*freq_time))
    x_cos = np.multiply(np.asmatrix(x),np.cos(2*np.pi*freq_time))
    sin_sum = np.sum(x_sin,1);
    cos_sum = np.sum(x_cos,1);

    mags = np.sqrt(np.multiply(sin_sum,sin_sum) + np.multiply(cos_sum,cos_sum))
    mags = np.squeeze(np.asarray(mags))/np.max(mags)
    #plt.stem(f_vect,mags)

    return [f_vect, mags, x, fs]
#############################################################################
from signal_processing import pure_tone_complex, sound, get_dft

harmonics = 12;

extract = extract_harmonics('instruments/cello_A4_normal.mp3', fs = 44100, f_0 = 440, n_harms = harmonics);
plt.figure(0)
plt.stem(extract[0],extract[1])

plt.figure(1)
dft = get_dft(extract[2],extract[3])
fig, (ax1,ax2) = plt.subplots(2,1,sharex = True)
ax1.plot(dft[0],dft[1]/np.max(dft[1]))
ax2.plot(dft[2])

plt.xlim([0,4500])
plt.show()


fs_Hz = 44.1e3;
dur_sec = 2;
amp = extract[1];
phi = np.zeros(harmonics);
freq_Hz = extract[0];

tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
sound(tone,fs_Hz,'violin_resynth.wav',1)
