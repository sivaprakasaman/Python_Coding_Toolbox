#Dependency: playsound
# TODO: figure out why sound doesn't play in atom. Should work when run from terminal though

from playsound import playsound
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
    playsound(fname)

    if not savefile:
        os.remove(fname)

################################################################################

#Implementation Example:

from signal_processing import pure_tone_complex

f0 = 200;

#Selected frequencies to generate a heavenly major chord with some spatial chara
#cteristics

freq_Hz = [f0,(5/4)*f0,(3/2)*f0];
freq_Hz2 = [5/4*f0,(3/2)*f0,5*f0];

fs_Hz = 44e3;
dur_sec = 2;
amp = [2,10,6];
phi = [0,0,0];

left = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)
right = pure_tone_complex(freq_Hz2, fs_Hz, dur_sec, amp, phi)

sound([left[1],right[1]],fs_Hz,'major_chord.wav',1)
