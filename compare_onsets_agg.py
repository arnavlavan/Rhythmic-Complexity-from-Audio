import librosa
import numpy as np
import matplotlib.pyplot as plt

music_file = '0044 zohar argov - nachon lehayom.mp3'
music_dir = 'data/mp3/'
img_dir = 'data/img/'

print('Loading ' + music_file + '...')
y, sr = librosa.load(music_dir + music_file, duration=3.0)
hop_length = 512
print("hpss...")
y_harmonic, y_percussive = librosa.effects.hpss(y)
# Beat track on the percussive signal
print("beat_frames...")
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
print('stft and times...')
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
print("onsets...")
oenv1 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
oenv2 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, fmax=8000, n_mels=256)
oenv3 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.median)
oenv4 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.median, fmax=8000,
                                    n_mels=256)
oenv5 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.max)
oenv6 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, feature=librosa.feature.chroma_stft)
oenv7 = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, feature=librosa.feature.mfcc)
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv1, backtrack=False)
onset_times = librosa.frames_to_time(onset_raw, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_count = np.arange(beat_frames.shape[0]) + 1
beats = {}
for i in np.arange(beat_frames.shape[0]):
    beats[beat_count[i]] = beat_times[i]
print(beats)

print('plotting...')

plt.figure(music_file.split('.mp3')[0] + ' excerpt', figsize=(8, 8))

ax2 = plt.subplot(211)
ax2 = plt.gca()
plt.title(music_file.split('.mp3')[0] + ' excerpt')
plt.margins(x=0.001, y=0.005)
# ax2.set_xticklabels(beats)
ax2.plot(times, 1 + oenv1/oenv1.max(), label='np.mean, default parameters')
ax2.plot(times, oenv2 / oenv2.max(), label='np.mean, fmax=8000, n_mels=256')
ax2.plot(times, 1 + oenv3 / oenv3.max(), alpha=0.75, label='np.median, default parameters')
ax2.plot(times, oenv4 / oenv4.max(), alpha=0.75, label='np.median, fmax=8000, n_mels=256')
plt.ylabel('normalized onset strength')
plt.legend()

ax3 = plt.subplot(212)
ax3 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax3.plot(times, oenv5 / oenv5.max(), label='np.max, default parameters')
ax3.plot(times, oenv6 / oenv6.max(), alpha=0.8, label='np.mean, feature=chroma_stft')
ax3.plot(times, oenv7 / oenv7.max(), alpha=0.6, label='np.mean, feature=mfcc')
plt.xlabel('time (s)')
plt.ylabel('normalized onset strength')
plt.legend()
plt.show()

'''
plt.figure(music_file.split('.mp3')[0] + ' excerpt')
plt.title(music_file.split('.mp3')[0] + ' excerpt')
plt.margins(x=0.001, y=0.005)
plt.plot(times, 1 + oenv2 / oenv2.max(), label='np.mean, fmax=8000, n_mels=256')
plt.plot(times, oenv4 / oenv4.max(), label='np.median, fmax=8000, n_mels=256')
plt.xlabel('time (s)')
plt.gca().set_yticklabels([])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
'''