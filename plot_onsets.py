import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('mesiba_behaifa_excerpt.mp3', duration=2.0)
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
onset_times = librosa.frames_to_time(onset_raw, sr=sr)


plt.figure()
plt.suptitle('0433 ultras & itay levi - messiba behaifa (excerpt)')

ax1 = plt.subplot(211)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power Spectrogram')
plt.xlabel('time (s)')

ax2 = plt.subplot(212)
ax2 = plt.gca()
plt.margins(x=0.001, y=0.005)
ax2.plot(times, oenv, label='Onset strength')
#ax2.spines['bottom'].set_visible(False)
#ax2.xaxis.set_ticks_position('bottom')
ax2.vlines(onset_times, 0, oenv.max(), linestyles = 'dotted', label='Onset time')
plt.title('Note Onsets')
plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.13), frameon=True, framealpha=0.75, fancybox=True)
plt.xlabel('time (s)')
ax2.xaxis.set_label_coords(0.05, -0.2)
vals = ax2.get_xticks()
ax2.set_xticklabels([str(i)[-2:] == '.0' and str(i)[:-2] or str(i) for i in vals])
plt.ylabel('normalized strength')

plt.subplots_adjust(hspace=0.55, right=0.95)
plt.show()
