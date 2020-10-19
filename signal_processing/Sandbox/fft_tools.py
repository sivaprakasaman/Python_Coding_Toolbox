#Written By: Andrew Sivaprakasam
#Last Updated: October, 2020

#DESCRIPTION: dft = get_dft(sig[], Fs, nfft, type, sides)
#Outputs dft[0] - frequency vector, dft[1] - 2-sided fft, dft[2] - phase

import numpy as np
import matplotlib.pyplot as plt
from signal_processing import pure_tone_complex

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
