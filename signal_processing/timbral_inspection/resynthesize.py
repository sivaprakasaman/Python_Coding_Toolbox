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


from signal_processing import pure_tone_complex, sound, magphase
import matplotlib.pyplot as plt
from playsound import playsound

## TODO: Whoops need to properly pass these params
def resynthesize(mags, fname = 'resynth.wav', freq_Hz = [0], dur_sec = 1, phi = [0], scale = .75, tone_shift = 1, env_fxn = 1, fs = 44100, type = 'sin', play_write = True, plot = True):
    harmonics = len(mags)

    #This handling should be added to pure_tone_complex at some point
    if len(phi)<harmonics:
        phi = np.ones(harmonics)*phi;

    if len(freq_Hz) <harmonics:
        freq_Hz = np.arange(1,n_harms+1)*440;

    tone = pure_tone_complex(freq_Hz*tone_shift, fs, dur_sec, mags, phi, type)
    tone = tone[1]*env_fxn;
    tone = scale*tone/np.max(tone);

    t_vect = np.arange(0,len(tone))/fs_Hz;

    if plot:
        plt.figure()
        plt.plot(tone);
        plt.xlim([0,len(tone)])

    if play_write:
        sound(tone,fs_Hz,fname,1)

    return tone

import numpy as np

def play_alma_mater(mags, freq_Hz, fname = 'alma_mater.wav', n_harms = 6,  key = 1, env_fxn = 1, type = 'sin'):
    shift_mat = [1.26/1.66, .85, .95, 1.00, 1.13, 1.26, 1.26, 1.32, 1.32, 1.32, 1, 1.13, 1.13, 1.26, 1.26/1.66, 1.26, 1.20, 1.26, 1.26, 1.13, 1.00, 1.13, 1.26, 1.26, 1.13, .85, .95, 1, .95, .85, 1.13, 1.26/1.66, 1.26/1.66, .85, .95, 1, 1.13, 1.26, 1.26, 1.26, 1.32, 1.32, 1, 1.13, 1.26, .85, .95, 1, .85, 1.26/1.66, 1, 1.26, 1.26/1.66, .85, 1.26, 1.13, 1, 1]
    dur_mat = [2, 1, 1, 1.5, .5, 1, 1, 1, .5, .5, 1, .5, .5, 1, 1, 1, 1, 2, 1, 1, 1.5, .5, 1, 1, 1, .5, .5, 1, .5, .5, 3, 1.5, .5, 1, 1, 1.5, .5, 1, .5, .5, 1, 1, 1, 1, 3, 1.5, .5, 1, 1, 1, 1, 1, 1, 1.5, .5, 1.5, .5, 3]
    scale_mat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1 , 1, 1, 1, 1]

    #Change tempo
    dur_mat = np.asarray(dur_mat)*.2

    tone = [];

    for i in range(0,len(shift_mat)):
        tone_temp = resynthesize(extract[1], freq_Hz  = key*freq_Hz, dur_sec = dur_mat[i], scale = scale_mat[i], tone_shift = shift_mat[i], env_fxn = env_fxn, type = type, play_write = True, plot = False)
        np.concatenate((tone, tone_temp), axis = 0)

    sound(tone, 44100, fname, 1)
########################## IMPLEMENTATION #####################################

from signal_processing import pure_tone_complex, sound, magphase
import matplotlib.pyplot as plt

#Can use the below line in Atom when running Hydrogen
#%matplotlib inline

harmonics = 6;

extract = extract_harmonics('instruments/bassoon_A4_normal.wav', fs = 44100, f_0 = 440, n_harms = harmonics);

fs_Hz = extract[3];
dur_sec = 1;
amp = extract[1];
phi = np.zeros(harmonics);
freq_Hz = extract[0];
#print(extract[1])

t_vect = np.arange(0,dur_sec*fs_Hz)/fs_Hz;
env_banj = np.exp(-9*t_vect);
env_string = (1+.1*np.sin(7*np.pi*2*t_vect))*np.sin(.5*np.pi*2*t_vect);

#tone = resynthesize(extract[1], 'resynthesize2.wav', freq_Hz = freq_Hz, dur_sec = 1, scale = .5, tone_shift = 1, env_fxn = env_banj, type = 'saw', play_write = True)

# plt.figure()
# plt.plot(tone)

fs, x  = wavfile.read('resynthesize2.wav')

plt.plot(x)

## TODO: get env function to pass through
play_alma_mater(extract[1], freq_Hz, key = .94)

# tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi,'sin')
# tone = tone[1]/np.max(tone[1]);
# t_vect = np.arange(0,len(tone))/fs_Hz;
# #tone = tone*np.exp(-9*np.arange(0,len(tone))/fs_Hz)
# tone = tone*(1+.1*np.sin(7*np.pi*2*t_vect))
# plt.figure(2)
# plt.plot(tone);
# plt.xlim([0,len(tone)])
# sound(tone,fs_Hz,'resynth.wav',1)
#
# ## TODO: Clean up plots, try to directly compare DFT to extracted harmonics
#
# plt.figure(0)
# plt.stem(extract[0],extract[1])
#
# plt.figure(1)
# fig, (ax1,ax2) = magphase(extract[2],extract[3],x_axislim = [0,np.max(extract[0])])
