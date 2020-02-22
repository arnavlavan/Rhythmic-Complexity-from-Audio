import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def norm_cc(a, b, start_idx, end_idx):
    res = np.zeros(end_idx - start_idx)
    b = (b - np.mean(b)) / (np.std(b))
    for i in range(start_idx, end_idx):
        vec_a = a[i:i+len(b)]
        # print( len(vec_a) , np.std(vec_a) , np.mean(vec_a))
        vec_a = (vec_a - np.mean(vec_a)) / (np.std(vec_a))
        if len(vec_a) < len(b):
            res[i - start_idx] = 0
        else:
            res[i - start_idx] = np.correlate(vec_a, b) / len(vec_a)
    return res


# main
music_file = '0433 ultras & itay levi - messiba behaifa.mp3'
music_dir = 'data/mp3/'
img_dir = 'data/img/'

print('Loading ' + music_file + '...')
y, sr = librosa.load(music_dir + music_file, duration=60.0)

yMax = np.max(y)
idxStart = (np.where(y[1:sr * 40] > (yMax / 100)))[0][0]
idxEnd = len(y) - sr*40 + (np.where(y[-sr*40:] > (yMax / 100)))[0][-1]
y = y[idxStart:idxEnd]
hop_length = 512

print("hpss...")
y_harmonic, y_percussive = librosa.effects.hpss(y)
# Beat track on the percussive signal
print("beat_frames...")
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

print('stft and times...')
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))

print("onset...")
oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.median, fmax=8000,
                                    n_mels=256)
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
onset_times = librosa.frames_to_time(onset_raw, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print("norm_cc...")
onset_cc_1 = np.zeros(int(np.ceil(len(beat_frames)/4)))
onset_cc_t = np.zeros(int(np.ceil(len(beat_frames)/4)))
cnt = 0

for beatIdx in range(0, len(beat_frames)-8, 4):
    if cnt%10 == 0:
        print(cnt)
    frameStart = beat_frames[beatIdx]
    frameEnd = beat_frames[beatIdx + 4]
    onSet_subVec = oenv[frameStart:frameEnd]
    ons1 = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx + 8])
    onset_cc_1[cnt] = np.max(ons1[1:])
    onset_cc_t[cnt] = frameStart*(hop_length/sr)
    cnt += 1

onset_cc_2 = np.zeros(int(np.ceil(len(beat_frames)/2)))
onset_cc_t2 = np.zeros(int(np.ceil(len(beat_frames)/2)))
cnt = 0

for beatIdx in range(0, len(beat_frames)-4, 2):
    if cnt%10 == 0:
        print(cnt)
    frameStart = beat_frames[beatIdx]
    frameEnd = beat_frames[beatIdx + 2]
    onSet_subVec = oenv[frameStart:frameEnd]
    ons2 = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx + 4])
    onset_cc_2[cnt] = np.max(ons2[1:])
    onset_cc_t2[cnt] = frameStart*(hop_length/sr)
    cnt += 1


print('plotting...')
plt.figure(music_file.split('.mp3')[0], figsize=(10, 8))
plt.suptitle(music_file.split('.mp3')[0])

ax1 = plt.subplot(211)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
# ax1.set_xlim(left=70, right=90)
plt.title('Power Spectrogram')
plt.xlabel('time (s)')

ax2 = plt.subplot(212)

ax2 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax2.plot(times, oenv, label='Onset strength', zorder=1, alpha=0.5)
plt.plot(onset_cc_t[np.where(onset_cc_t > 0)], onset_cc_1[np.where(onset_cc_t > 0)],
         label='onset cross-correlation (bar segment)', zorder=2, alpha=0.75)
plt.plot(onset_cc_t2[np.where(onset_cc_t2 > 0)], onset_cc_2[np.where(onset_cc_t2 > 0)],
         label='onset cross-correlation (half bar segment)', zorder=4)
plt.vlines(beat_times[0::4], 0, 1, linestyle='--', label='bars', zorder=3, alpha=0.6)
plt.title(music_file.split('.mp3')[0] + ' excerpt')
plt.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.75, fancybox=True)
plt.xlabel('time (s)')
# ax2.set_xlim(left=ax1.get_xlim()[0], right=ax1.get_xlim()[1])
vals = ax2.get_xticks()
ax2.set_xticklabels([str(i)[-2:] == '.0' and str(i)[:-2] or str(i) for i in vals])
plt.ylabel('normalized strength/CC')

'''
ax3 = plt.subplot(313)
ax3 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax3.plot(times, oenv, label='Onset strength')
plt.plot(onset_cc_t[np.where(onset_cc_t > 0)], onset_cc_2[np.where(onset_cc_t > 0)],
         label='Onset cross-correlation')
plt.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.75, fancybox=True)
plt.xlabel('time (s)')
vals_1 = ax3.get_xticks()
ax3.set_xticklabels([str(i)[-2:] == '.0' and str(i)[:-2] or str(i) for i in vals_1])
plt.ylabel('Cross-correlation values')

ax3 = plt.subplot(313)
plt.margins(x=0.001, y=0.005)
ax3.plot(times, oenv_norm, label='Normalized Onsets Strength')
plt.legend(loc='lower center', ncol=2, frameon=True, framealpha=0.75, fancybox=True)
'''
plt.subplots_adjust(hspace=0.55, right = 0.975, left = 0.06)
plt.show()
print('saving plot...')
img_file = music_file.split('.mp3')[0] + '.png'
# plt.savefig(img_dir + img_file)
plt.close()

#bbox_to_anchor=(0.38, 1.2)