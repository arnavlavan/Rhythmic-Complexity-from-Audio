import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def norm_vec(vec):
    res = (vec - np.mean(vec)) / (np.std(vec))
    return res


music_file = '0433 ultras & itay levi - messiba behaifa.mp3'
music_dir = 'data/mp3/'
img_dir = 'data/img/'

print('Loading ' + music_file + '...')
y, sr = librosa.load(music_dir + music_file, duration=60.0)
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
beats = np.arange(beat_frames.shape[0]) + 1
print(beats)
print(beat_times)

a = 58
b = a + 4
c = a + 8
segment_a = oenv[beat_frames[a]:beat_frames[b]]
segment_b = oenv[beat_frames[b]:beat_frames[c]]
norm_a = norm_vec(segment_a)
norm_b = norm_vec(segment_b)
beats_a = beats[a-1:b+1]
beats_b = beats[b-1:c+1]


# print(np.corrcoef(segment_a, norm_a) + ' correlation coefficient a')
# print(np.corrcoef(segment_b, norm_b) + ' correlation coefficient b')

print('plotting...')
plt.figure(music_file.split('.mp3')[0] + ' excerpts', figsize=(8, 12))
# plt.suptitle(music_file.split('.mp3')[0] + ' excerpts')

ax1 = plt.subplot(321)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='s')
ax1.set_xlim(left=beat_times[a], right=beat_times[b])
plt.title('Segment A')
plt.xlabel('time (s)')

ax2 = plt.subplot(322)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='s')
ax2.set_xlim(left=beat_times[b], right=beat_times[c])
plt.title('Segment B')
plt.xlabel('time (s)')

ax3 = plt.subplot(323)
ax3 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax3.set_xticklabels(beats_a)
# plt.xticks(np.arange(0, 1, 0.2), beats_a)
ax3.plot(segment_a, label='Onset Strength')
plt.legend()

ax4 = plt.subplot(324)
ax4 = plt.gca()
plt.margins(x=0.001, y=0.005)
# plt.xticks(np.arange(0, 1, 0.2), beats_b)
ax4.set_xticklabels(beats_b)
ax4.plot(segment_b, label='Onset Strength')
plt.legend()

ax5 = plt.subplot(325)
ax5 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax5.plot(norm_a, label='Normalized Onset Strength')
ax5.set_xticklabels(beats_a)
plt.xlabel('Beats')
plt.legend()

ax6 = plt.subplot(326)
ax6 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax6.plot(norm_b, label='Normalized Onset Strength')
ax6.set_xticklabels(beats_b)
plt.xlabel('Beats')
plt.legend()

plt.tight_layout()
plt.show()
