#Written By: Andrew Sivaprakasam
#Last Updated: October, 2020

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

###############################################################################

#Implemenation Example:

from signal_processing import pure_tone_complex

freq_Hz = [100,200];
fs_Hz = 44e3;
dur_sec = 2;
amp = [1,2];
phi = [0,0];
tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
fig, (ax1,ax2) = magphase(tone[1], fs_Hz, x_axislim = [0,400])
