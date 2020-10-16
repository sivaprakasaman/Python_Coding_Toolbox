#Last Edited: October, 2020 | Andrew Sivaprakasam
#Notes: sounddevice may be imported if you want to play tone, but throwing an
#ALSA error on unix rn.


#imports needed
import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
#Creating the Function:
def pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi):
    #Returns an 1x2 array with signal and time
    #probably would be good to add some checks here
    samples = np.arange(0,fs_Hz*dur_sec)
    time_sec = samples/fs_Hz
    #Initializing Signal
    sig = [];

    for i in range(0,len(freq_Hz)):
        sig_temp = amp[i]*np.sin(2*np.pi*freq_Hz[i]*time_sec + phi[i])
        sig_arr = np.array(sig_temp)

        if i == 0:
            sig = np.array(sig_temp)
            sig_total = sig
        else:
            sig = np.vstack((sig,sig_arr))
            sig_total = sig.sum(axis = 0)

    return([time_sec, sig_total])

#running function:
freq_Hz = [100,200];
fs_Hz = 44e3;
dur_sec = 1;
amp = [1,1];
phi = [0,0];

tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)

plt.plot(tone[0],tone[1])
plt.xlim(0,0.01)

plt.show

#sd.play(tone[1],fs_Hz)
