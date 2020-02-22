import matplotlib.pyplot as plt
import librosa

y, sr = librosa.load('data/mp3/0015 peer tassi - dfikot halev.mp3')
librosa.feature.chroma_stft(y=y, sr=sr)


plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()