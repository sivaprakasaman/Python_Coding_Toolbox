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
    #output = get_spect(x, fs, DR = 120, BW = 100, xlim = [0,0.5], ylim = [0,5000], colormap = 'magma')

    ## TODO: Try applying dpss to this. Might result in more accurate
    ## magnitudes?

    freq_time = np.multiply(np.asmatrix(f_vect).T,np.asmatrix(t_vect))
    x_sin = np.multiply(np.asmatrix(x),np.sin(2*np.pi*freq_time))
    x_cos = np.multiply(np.asmatrix(x),np.cos(2*np.pi*freq_time))
    sin_sum = np.sum(x_sin,1);
    cos_sum = np.sum(x_cos,1);

    mags = np.sqrt(np.multiply(sin_sum,sin_sum) + np.multiply(cos_sum,cos_sum))
    mags = np.squeeze(np.asarray(mags))/np.max(mags)

    phase = np.arctan(np.divide(sin_sum,cos_sum));
    phase = np.squeeze(np.asarray(phase));
    #phase = [0];
    #plt.stem(f_vect,mags)

    return [f_vect, mags, phase, x, fs]

from signal_processing import pure_tone_complex, sound, magphase
import matplotlib.pyplot as plt
#from playsound import playsound

def resynthesize(mags, fname = 'resynth.wav', fs_Hz = 44100, freq_Hz = [0], dur_sec = 1, phi = [0], scale = .75, tone_shift = 1, env_fxn = 1, fs = 44100, type = 'sin', play_write = True, plot = True):
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
################################################################################

import numpy as np

def play_alma_mater(extract, freq_Hz, fname = 'alma_mater.wav', n_harms = 6,  key = 1, tempo = 0.3, fxn = 'string', type = 'sin', short = True):
    shift_mat = [1.26/1.66, .85, .95, 1.00, 1.13, 1.26, 1.26, 1.32, 1.32, 1.32, 1, 1.13, 1.13, 1.26, 1.26/1.66, 1.26, 1.20, 1.26, 1.26, 1.13, 1.00, 1.13, 1.26, 1.26, 1.13, .85, .95, 1, .95, .85, 1.13, 1.26/1.66, 1.26/1.66, .85, .95, 1, 1.13, 1.26, 1.26, 1.26, 1.32, 1.32, 1, 1.13, 1.26, .85, .95, 1, .85, 1.26/1.66, 1, 1.26, 1.26/1.66, .85, 1.26, 1.13, 1, 1]
    dur_mat = [2, 1, 1, 1.5, .5, 1, 1, 1, .5, .5, 1, .5, .5, 1, 1, 1, 1, 2, 1, 1, 1.5, .5, 1, 1, 1, .5, .5, 1, .5, .5, 3, 1.5, .5, 1, 1, 1.5, .5, 1, .5, .5, 1, 1, 1, 1, 4, 1.5, .5, 1, 1, 1, 1, 1, 1, 1.5, .5, 1.5, .5, 3]
    scale_mat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1 , 1, 1, 1, 1]

    #Truncate by default, otherwise listen to music for a few extra seconds...
    if short:
        shift_mat = shift_mat[:6];
        dur_mat = dur_mat[:6];
        scale_mat = scale_mat[:6];

    fs = 44100;
    #Change tempo
    dur_mat = np.asarray(dur_mat)*tempo
    tone = [];

    for i in range(0,len(shift_mat)):

        t_vect = np.arange(0,dur_mat[i]*fs)/fs;

        if fxn == 'banjo':
            env_fxn = np.exp(-7*t_vect);
        elif fxn == 'string':
            env_fxn = (1+.25*np.sin(5*np.pi*2*t_vect))*np.sin(.5*np.pi*2*t_vect);
        else:
            env_fxn = 1;

        tone_temp = resynthesize(extract[1], freq_Hz  = key*freq_Hz, dur_sec = dur_mat[i], phi = extract[2], scale = scale_mat[i], tone_shift = shift_mat[i], env_fxn = env_fxn, type = type, play_write = False, plot = False)
        print(tone_temp)
        tone = np.concatenate((tone,tone_temp), axis = 0)

    sound(tone, fs, fname, 1)

    return [tone,fs];

########################## IMPLEMENTATION #####################################

# from signal_processing import pure_tone_complex, sound, magphase, get_spect
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram as sp
# import numpy as np
# ## TODO: Quantify Envelope, apply slepian sequences, verify magnitudes against DFT/PSD

# #Can use the below line in Atom when running Hydrogen
# #%matplotlib inline

# harmonics = 7;
# first = 0;
# dur_sec = 1;
# toPlay = np.array([0,1,2,3,4,5,6])
# extract = extract_harmonics('instruments/violin_A4_normal.wav', fs = 44100, f_0 = 440, n_harms = harmonics);

# fs_Hz = extract[4];
# amp = extract[1][toPlay];
# phase = extract[2][toPlay];
# freq_Hz = extract[0][toPlay];

# t_vect = np.arange(0,dur_sec*fs_Hz)/fs_Hz;
# env_banj = np.exp(-9*t_vect);
# env_string = (1+0.15*np.sin(6*np.pi*2*t_vect))*np.sin(.5*np.pi*2*t_vect);

# tone = resynthesize(amp, 'violin_all.wav', freq_Hz = freq_Hz, dur_sec = 1, phi = phase, scale = 1, tone_shift = 1, env_fxn = env_string, type = 'sin', play_write = True, plot = False)

# sound(tone, fs_Hz)
# get_spect(tone, fs_Hz, DR = 200, BW = 75, xlim = [0,1], ylim = [0,4000], colormap = 'cividis',title = 'Simulated Violin | All Harmonics');

# #Play Alma Mater
# alma_mater = play_alma_mater(extract, freq_Hz, key = 1, fxn = 'strings', type = 'sin')
#
# plt.figure()
# plt.plot(np.arange(0,len(alma_mater[0]))/alma_mater[1],alma_mater[0]);
# output = get_spect(alma_mater[0],alma_mater[1], DR = 300, BW = 200, xlim = [0.01,2], ylim = [0,5000])
