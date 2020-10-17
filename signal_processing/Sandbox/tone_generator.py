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

#Implementation Example:
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
