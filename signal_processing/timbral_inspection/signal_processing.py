#Written By: Andrew Sivaprakasam
#Last Updated: October, 2020

#Here are the currently working signal processing scripts written in this folder. All un-verified
#scripts have been worked out in the 'Sandbox' directory.

#DEPENDENCIES (make sure to install these python packages):
# -numpy | pip install numpy
# -scipy | pip install scipy
# -matplotlib (only if you are plotting) | pip install matplotlib


#To implement these scripts in another script, make sure to have the 'signal_processing.py'
#script in the directory, and call functions by importing from this file:

#Example:
#from signal_processing import pure_tone_complex, get_dft

###############################################################################

#DESCRIPTION: tone = pure_tone_complex(freq_Hz[], fs_Hz, dur_sec, amp[], phi[])

#Outputs a tone with frequencies in array freq_Hz, with corresponding amplitdues/phases in amp/phi, and its time vector
#REMEMBER! The output is an ARRAY, with tone[0] being a time vector, and tone[1] being the signal


#imports needed
import numpy as np
import matplotlib.pyplot as plt

#Creating the Function:
def pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi):

    samples = np.arange(0,fs_Hz*dur_sec)
    time_sec = samples/fs_Hz

    #Initializing Signal
    sig = [];

    for i in range(0,len(freq_Hz)):
        sig_temp = amp[i]*np.sin(2*np.pi*freq_Hz[i]*time_sec + phi[i])
        sig_arr = np.array(sig_temp)

        if i == 0:
            sig_total =  sig_arr
        else:
            sig_total = sig_total + sig_arr

    return([time_sec, sig_total])

#Implementation Example (uncomment to run):
# freq_Hz = [100,100];
# fs_Hz = 44e3;
# dur_sec = 1;
# amp = [1,1];
# phi = [0,np.pi];
#
# tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
#
# plt.plot(tone[0],tone[1])
# plt.xlim(0,0.01)
#
# plt.show()

#sd.play(tone[1],fs_Hz)

################################################################################

#DESCRIPTION: get_dft(sig[], Fs, nfft, type)

#Outputs dft[0] - frequency vector, dft[1] - 2-sided fft, dft[2] - phase

import numpy as np
import matplotlib.pyplot as plt

def get_dft(sig, fs, nfft = 0, type = 'mag'):

    if nfft < np.power(2,np.ceil(np.log2(len(sig)))):
        nfft = np.power(2,np.ceil(np.log2(len(sig))))

    fft_c = np.fft.fftshift(np.fft.fft(sig,np.int(nfft)))
    fft_n = np.absolute(fft_c)/fs
    freq = np.fft.fftshift(np.fft.fftfreq(fft_n.shape[-1],1/fs))
    phase = np.angle(fft_c, deg = True)

    if type == 'dB':
        fft_db = 20*(np.log10(fft_n))
        return [freq, fft_db, phase]
    elif type == 'complex':
        return [freq, fft_c, phase]
    else:
        return [freq, fft_n, phase]

#Implementation of get_fft fxn

#
# freq_Hz = [100,200];
# fs_Hz = 44e3;
# dur_sec = 2;
# amp = [1,2];
# phi = [0,0];
# tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
#
# dft = get_dft(tone[1],fs_Hz)
#
# fig, (ax1,ax2) = plt.subplots(2,1,sharex = True)
# ax1.plot(dft[0],dft[1])
# ax2.plot(dft[2])
#
# plt.xlim([0,400])
#
# #Below line suggested for dB, otherwise an artifact appears at <-200db
# #ax1.set_ylim([-100,np.max((dft[1]))+10])
#
# ax1.set_ylabel('Magnitude')
# ax2.set_ylabel('Phase (Deg)')
# plt.xlabel('Frequency (Hz)')
# ax1.set_title('Example Output')
# plt.show()
#
# plt.figure(2)
# plt.plot(tone[0],tone[1])
# plt.xlim([0,1/freq_Hz[0]])

###############################################################################
#DESCRIPTION: sound(sig[], fs, fname (optional), savefile (optional))

#Plays sig through computer audio output. Use 2 dimensions for sig if you want
#stereo output, if 1 dimension, sound() assumes mono. fname is the filename you
#to save as, (default = 'sound.wav'), savefile = 1 if you want to save the wav
#file or savefile = 0 if you want it to autodelete

import numpy as np
import os
from scipy.io.wavfile import write

#sig should have one dim for mono or two dims for stereo dim = channel numbers
def sound(sig, fs, fname = 'sound.wav', savefile = 0):

    sigf32 = np.float32(sig)
    fs = int(fs)

    sigf32 = sigf32/np.max(sigf32)
    write(fname,fs,sigf32.T)

    pwd = os.getcwd()
    wav_file = pwd + '/' + fname
    os.system('aplay ' + fname)

    if not savefile:
        os.remove(fname)

################################################################################

#Implementation Example:

# from signal_processing import pure_tone_complex
#
# f0 = 200;
#
# #Selected frequencies to generate a heavenly major chord with some spatial chara
# cteristics
#
# freq_Hz = [f0,(5/4)*f0,(3/2)*f0];
# freq_Hz2 = [5/4*f0,(3/2)*f0,5*f0];
#
# fs_Hz = 44e3;
# dur_sec = 2;
# amp = [2,10,6];
# phi = [0,0,0];
#
# left = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
# right = pure_tone_complex(freq_Hz2, fs_Hz, dur_sec, amp, phi)
#
# sound([left[1],right[1]],fs_Hz,'major_chord.wav',1)

###############################################################################
#DESCRIPTION: fig, (ax1,ax2) = magphase(sig,fs,nfft = 0,type = 'mag', axis = 'linear', x_axislim = [0,0])

#This script is nice way to visualize the magnitude and phase of a signal. type corresponds to what FFT representation is desired
#-['mag', 'dB', 'complex']. axis can be 'linear' or 'log'. x_axislim[] is just what limits you want on x axis.

#Not sure generally how useful phase from the DFT is for
#anything other than impulse response, but will probably learn. if not useful, will remove the phase subplot.

import matplotlib.pyplot as plt
import numpy as np
from signal_processing import get_dft

def magphase(sig,fs,nfft = 0,type = 'mag', axis = 'linear', x_axislim = [0,0]):

    if x_axislim[1]-x_axislim[0] <= 0:
        x_axislim[0] = -fs/2
        x_axislim[1] = fs/2

    dft = get_dft(sig,fs,nfft,type)

    fig, (ax1,ax2) = plt.subplots(2,1,sharex = True)
    if axis == 'log':
        ax1.semilogx(dft[0],dft[1])
        ax2.semilogx(dft[0],dft[2])
    else:
        ax1.plot((dft[0]),dft[1])
        ax2.plot(dft[0],dft[2])

    ax1.set_xlim([x_axislim[0],x_axislim[1]])
    ax1.set_ylabel('Magnitude')
    ax2.set_ylabel('Phase (Deg)')
    plt.xlabel('Frequency (Hz)')

    ax1.set_title('Magnitude/Phase' + ' - ' + type)
    plt.show()

    return fig, (ax1,ax2)

########################## IMPLEMENTATION #####################################

#Implemenation Example:

# from signal_processing import pure_tone_complex
#
# freq_Hz = [100,200];
# fs_Hz = 44e3;
# dur_sec = 2;
# amp = [1,2];
# phi = [0,0];
# tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
# fig, (ax1,ax2) = magphase(tone[1], fs_Hz, x_axislim = [0,400])
