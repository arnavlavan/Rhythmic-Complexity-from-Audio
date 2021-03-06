import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import dill
import os


def norm_cc(a,b,startIdx,endIdx):
    res = np.zeros(endIdx - startIdx)
    b = (b - np.mean(b)) / (np.std(b))
    for i in range(startIdx,endIdx):
        vec_a = a[i:i+len(b)]
        #print( len(vec_a) , np.std(vec_a) , np.mean(vec_a))
        vec_a = (vec_a - np.mean(vec_a)) / (np.std(vec_a))
        if len(vec_a) < len(b):
            res[i - startIdx] = 0
        else:
            res[i - startIdx] = np.correlate(vec_a,b) / len(vec_a)
    return res


### main
filename = "0075 pe'er tassi - derech hashalom.mp3"
filepath = '../../data/mp3/'
data_dir = '../../data/pkl/'
data_file = filename.split('.mp3')[0] + '.dill'
'''
print('Loading ...')
y, sr = librosa.load(filepath + filename)

yMax = np.max(y)
idxStart = (np.where(y[1:sr * 40] > (yMax / 100)))[0][0]
idxEnd = len(y) - sr*40 + (np.where(y[-sr*40:] > (yMax / 100) ))[0][-1]
y = y[idxStart:idxEnd]
hop_length = 512

print("hpss...")
y_harmonic, y_percussive = librosa.effects.hpss(y)
# Beat track on the percussive signal
print("beat_frames...")
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

print('stft and times...')
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))

print("onset...")
oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.median, fmax=8000,
                                    n_mels=256)
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
onset_times = librosa.frames_to_time(onset_raw, sr=sr)

print("norm_cc...")
onset_cc = np.zeros(int(np.ceil(len(beat_frames)/4)))
onset_cc_t = np.zeros(int(np.ceil(len(beat_frames)/4)))
cnt = 0
for beatIdx in range(0,len(beat_frames)-8,4):
    if cnt%10 == 0:
        print(cnt)
    frameStart = beat_frames[beatIdx]
    frameEnd = beat_frames[beatIdx + 4]
    onSet_subVec = oenv[frameStart:frameEnd]
    res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx + 8])
    onset_cc[cnt] = np.max(res[1:])
    onset_cc_t[cnt] = frameStart*(hop_length/sr)
    cnt += 1

dill.dump_session(data_dir + data_file)

print('plotting...')
plt.figure(data_file, figsize=(10,6))
plt.suptitle(data_file)


ax1 = plt.subplot(211)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
#ax1.set_xlim(left=70, right=90)
plt.title('Power Spectrogram')
plt.xlabel('time (s)')


ax2 = plt.subplot(212)'''

dill.load_session(data_dir + data_file)

tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0]/2)
ac_global = librosa.util.normalize(ac_global)

plt.figure(data_file, figsize=(10,6))
plt.suptitle(data_file)
ax2 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax2.plot(times, oenv, label='Onset strength')
plt.plot(onset_cc_t[np.where(onset_cc_t > 0)],onset_cc[np.where(onset_cc_t > 0)], label='Onset cross-correlation')
plt.plot(ac_global, label = 'Onset global auto-correlation')
plt.title('Note Onsets')
#plt.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.75, fancybox=True)
plt.xlabel('time (s)')
#ax2.set_xlim(left=ax1.get_xlim()[0], right=ax1.get_xlim()[1])
vals = ax2.get_xticks()
ax2.set_xticklabels([str(i)[-2:] == '.0' and str(i)[:-2] or str(i) for i in vals])
plt.ylabel('normalized strength/CC')
plt.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.75, fancybox=True)

plt.subplots_adjust(hspace=0.55, right = 0.975, left = 0.06)
plt.show()
