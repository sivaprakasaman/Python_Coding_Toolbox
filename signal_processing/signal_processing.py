#Written By: Andrew Sivaprakasam
#Last Updated: October, 2020

#DESCRIPTION: tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
#Outputs a tone with frequencies in array freq_Hz, with corresponding amplitdues/phases in amp/phi, and its time vector
#REMEMBER! The output is an ARRAY, with tone[0] being a time vector, and tone[1] being the signal

#imports needed
import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd

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
# plt.show

#sd.play(tone[1],fs_Hz)

################################################################################

#DESCRIPTION: get_dft(sig[], Fs, nfft, type, sides)
#Outputs a matrix with index 0 - frequency vector, 1 - fft, 2 - phase

import numpy as np
import matplotlib.pyplot as plt
from tone_generator import pure_tone_complex

def get_dft(sig, fs, nfft = 0, type = 'mag'):

    if nfft < np.power(2,np.ceil(np.log2(len(sig)))):
        nfft = np.power(2,np.ceil(np.log2(len(sig))))

    fft_c = np.fft.fft(sig,np.int(nfft))
    fft_n = np.absolute(fft_c)/fs
    freq = np.fft.fftfreq(fft_n.size,1/fs)
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
