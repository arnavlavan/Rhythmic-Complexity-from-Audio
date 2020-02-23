import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('../mp3/0014 moshe peretz - kol hamilim hasmechot.mp3')
hop_length = 512
y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
print(tempo)
beat_frames = np.append(beat_frames, 277)
beat_frames = np.append(beat_frames, 316)

print(beat_frames)
oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length, aggregate=np.median, fmax=8000,
                                    n_mels=256)
# D = np.abs(librosa.stft(y))
# times = librosa.frames_to_time(np.arange(D.shape[1]))

title_name = '0014 moshe peretz - kol hamilim hasmechot excerpt 01:19-01:27'
plt.figure(title_name, figsize=(10,4))
plt.title(title_name)
times = librosa.frames_to_time(np.arange(len(oenv)),sr=sr, hop_length=hop_length)
beats = times[beat_frames]
plt.plot(times, librosa.util.normalize(oenv), alpha=0.75, label='onset strength')
plt.vlines(beats, 0, 1, color='r', linestyle='--', label='beats')
plt.legend(frameon=True, framealpha=0.75)
ax = plt.gca()
ax.set_xticks(times[beat_frames])
labels = ['[1',2,3,4,5,6,7,8,']9']
ax.set_xticklabels(labels)
plt.ylabel('normalized strength')
plt.xlabel('beat count')

# plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
plt.tight_layout()
plt.show()