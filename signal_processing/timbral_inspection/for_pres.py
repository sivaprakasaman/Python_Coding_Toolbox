from signal_processing import pure_tone_complex, sound, magphase, get_spect
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp
import scipy as scip
import numpy as np

fs = 44100
fc = 4e3;
freq_Hz = [440];
dur_sec = 1;
mags = [1];
phi = [0];

F_0 = pure_tone_complex(freq_Hz, fs, dur_sec, mags, phi)
env = np.less(0,F_0[1])*F_0[1];


car = pure_tone_complex([fc],fs,dur_sec,mags,phi);
sos = scip.signal.butter(4,.2*fc,'low',fs = fs, output = 'sos');
env = scip.signal.sosfilt(sos,env);


stim = env*car[1];

plt.figure()
plt.plot(F_0[0],F_0[1])
plt.xlim([0/440,5/440])
plt.title('Pure Tone')

plt.figure()
plt.plot(F_0[0],stim)
plt.xlim([0/440,5/440])
plt.title('Transposed Tone')

# sound(stim,fs,fname = 'transposed.wav',savefile = 1)
# sound(.5*F_0[1],fs,fname = 'pure_440.wav',savefile = 1)

get_spect(stim, fs, DR = 220, BW = 80, xlim = [0,1], ylim = [0,8e3], colormap = 'cividis', title = 'Spectrogram | Transposed Tone');
