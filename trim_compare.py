import librosa
import os
import numpy as np

file_path = 'C:/Users/OWNER/Dropbox/1phd/database/files/'

for filename in os.listdir(file_path):
    allData = {}
    print(filename)
    y, sr = librosa.load(file_path + filename)

    yMax = np.max(y)
    idxStart = (np.where(y[1:sr * 40] > (yMax / 60)))[0][0]
    idxEnd = len(y) - sr*40 + (np.where(y[-sr*40:] > (yMax / 60)))[0][-1]
    y_manual = y[idxStart:idxEnd]
    y_trim, index = librosa.effects.trim(y)
    print('original track: ' + str(librosa.get_duration(y)))
    print('manual trimming: ' + str(librosa.get_duration(y_manual)))
    print('librosa trimming: ' + str(librosa.get_duration(y_trim)))
