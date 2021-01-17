#Andrew Sivaprakasam
#Purdue University
#Email: asivapr@purdue.edu

#DESCRIPTION: Code written to quickly convert mp3 files to wav. Trying this
#because librosa was having some issues


#Additional Dependencies: librosa, numpy, matplotlib
# pip3 install librosa
# pip3 install numpy
# pip3 install matplotlib

from pydub import AudioSegment

def mp3_to_wav(fname):
    sound = AudioSegment.from_mp3(fname+".mp3")
    sound.export(fname+".wav",format ="wav")

######################### IMPLEMENTATION ###################################

import os

arr = os.listdir("instruments/")
for i in range(1,len(arr)):
    mp3_to_wav("instruments/" + arr[i][0:-4])
