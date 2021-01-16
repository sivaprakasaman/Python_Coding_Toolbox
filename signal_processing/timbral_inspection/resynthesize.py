import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import matplotlib.pyplot as plt
import os

def extract_harmonics(fname, fs = 44100, f_0 = 440, n_harms = 6):
    aud  = lb.load(fname)
    x = np.array(aud[0])
    t_vect = np.arange(0,len(x))/fs
    f_vect = np.arange(1,n_harms+1)*f_0;
    #comb = np.mat(f_vect).T*np.mat(t_vect);
    comb = np.multiply(np.asmatrix(f_vect).T,np.asmatrix(t_vect))
    #print(comb)
    #x_sin = np.mat(x).T*np.mat(np.sin(2*np.pi*comb))
    x_sin = np.multiply(np.asmatrix(x),np.sin(2*np.pi*comb))
    x_cos = np.multiply(np.asmatrix(x),np.cos(2*np.pi*comb))
    sin_sum = np.sum(x_sin,1);
    cos_sum = np.sum(x_cos,1);
    #sin1 = np.squeeze(np.asarray(x_sin[2]))
    #plt.plot(t_vect,sin1)

    mags = np.sqrt(np.multiply(sin_sum,sin_sum) + np.multiply(cos_sum,cos_sum))
    plt.plot(f_vect,np.squeeze(np.asarray(mags)))


#############################################################################

extract_harmonics('instruments/viola_A4_normal.mp3');
