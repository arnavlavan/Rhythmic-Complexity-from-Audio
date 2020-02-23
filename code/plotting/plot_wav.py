import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('data/mp3/0433 ultras & itay levi - messiba behaifa.mp3')
y_harmonic, y_percussive = librosa.effects.hpss(y)

plt.figure(1, figsize=(15, 5))
librosa.display.waveplot(y_harmonic, sr=sr, alpha=0.2)
librosa.display.waveplot(y_percussive, sr=sr, color='r', alpha=0.3)
plt.tight_layout()
plt.show()
