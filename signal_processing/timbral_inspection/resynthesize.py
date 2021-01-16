import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import matplotlib.pyplot as plt
import os

def extract_harmonics(fname, fs = 44100, f_0 = 440, n_harms = 3):
    x  = lb.load(fname)
    plt.plot(np.arange(0,len(x[0])), x[0])
    f_vect = f_0*np.arange(1,n_harms+1);
    t_vect = np.arange(0,len(x[0]))/fs;
    temp = np.mat(f_vect).T*np.mat(t_vect);

    x_sin = np.multiply(np.mat(x[0]),np.mat(np.sin(2*np.pi*temp)))
    print((t_vect.shape))
    plt.plot(t_vect,x_sin[0])
#############################################################################

extract_harmonics('instruments/violin_A4_normal.mp3');
