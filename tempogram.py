import librosa
import numpy as np, scipy, matplotlib.pyplot as plt

# Load audio
filename = "data/mp3/0377 sarit hadad - tishtok tishtok.mp3"
x, sr = librosa.load(filename, duration=15.0)
print("decoded " + filename)

# compute onset envelope
hop_length = 200 # samples per frame
onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)
print("computed onset " + filename)

tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
ac_global = librosa.autocorrelate(onset_env, max_size=tempogram.shape[0])
ac_global = librosa.util.normalize(ac_global)

plt.figure(1)
ax = plt.gca()
#plt.plot(np.mean(tempogram, axis=1), label='Mean local autocorrelation')
plt.plot(ac_global, label='Global autocorrelation')
plt.legend(frameon=True)
#plt.xlabel('Lag (seconds)')
#vals = ax.get_xticks()
#ax.set_xticklabels([str('{0:.2f}'.format((i * float(hop_length) / sr))) for i in vals])
plt.show()

'''
x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr, num=tempogram.shape[0])
plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
plt.xlabel('Lag (seconds)')
plt.axis('tight')
plt.legend(frameon=True)
plt.show()
'''