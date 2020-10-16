import numpy as np
import matplotlib.pyplot as plt
from tone_generator import pure_tone_complex

def get_dft(sig, fs, nfft, type):
    fft_c = np.fft.fft(sig)
    fft_n = np.absolute(fft_c)
    freq = np.fft.fftfreq(fft_n.size,1/fs)
    return [freq,fft_n]

#Implementation of get_fft fxn
freq_Hz = [100];
fs_Hz = 44e3;
dur_sec = 1;
amp = [1];
phi = [0];

tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
dft = get_dft(tone[1],fs_Hz,None,'None')

plt.plot(dft[0],dft[1])
plt.xlim([0,200])
