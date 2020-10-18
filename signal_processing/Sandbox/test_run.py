from signal_processing import pure_tone_complex, get_dft, sound
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib - use this or above to plot inline or in separate figure

freq_Hz = [100,200];
fs_Hz = 44e3;
dur_sec = 1;
amp = [1,1];
phi = [0,0];

tone = pure_tone_complex(freq_Hz, fs_Hz, dur_sec, amp, phi)

sound(tone,fs_Hz)

dft = get_dft(tone[1], fs_Hz)


plt.plot(tone[0],tone[1])
plt.xlim(0,0.05)

plt.figure(2)
plt.plot(dft[0],dft[1])
plt.xlim([0,300])
plt.show()
