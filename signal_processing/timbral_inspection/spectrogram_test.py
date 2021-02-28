import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as sp

def get_spect(sig, Fs, BW = 0, DR = 0, ylim = [], xlim = [], Window = 'hamming', shading = 'goraud', colormap = 'viridis', title = 'Spectrogram', ytitle = 'Frequency (Hz)', xtitle = 'Time (s)'):

    if BW == 0:
        BW = 25;
    if DR == 0:
        DR = 30;


    Nwindow = np.round(4*Fs/BW);
    PercentOverlap = 90;
    OverlapFactor = (100-PercentOverlap)/100 + 1;
    Noverlap = np.round(Nwindow/OverlapFactor);
    Nfft = max(256,2**(np.floor(np.log2(Nwindow))+1));

    f, t, Sgram = sp(sig,Fs, window = Window, nperseg = int(Nwindow), noverlap = int(Noverlap), nfft = int(Nfft));

    SgramFactor = 10**-(DR/20);
    SgramFloor = SgramFactor*np.argmax(abs(Sgram));
    Sgram_dB = 20*np.log10(abs(Sgram)+SgramFloor);

    if len(ylim) == 0:
        ylim = [0,22000];
    if len(xlim) == 0:
        xlim = [0,max(t)];

    plt.figure()
    plt.pcolormesh(t,f,Sgram_dB, shading = shading, cmap = colormap);
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.show()

##########################IMPLEMENTATION EXAMPLE:##############################
from signal_processing import sound

#Example from SciPy.org
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 50*np.cos(2*np.pi*1*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

#sound(x,fs);
get_spect(x,fs, BW = 210, DR = 200, ylim = [0,5000], colormap = 'magma')
