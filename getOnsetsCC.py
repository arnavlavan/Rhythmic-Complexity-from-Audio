import librosa
import numpy as np
# from matplotlib import __version__  as matplotlib__version__
import matplotlib.pyplot as plt
import pickle
import os

# np.seterr(all='raise')
filepath = './data/mp3/'
filepath_pkl = 'data/pkl/'
filepath_img = 'data/img/'
PLOT = True


def save2data(_str, data2save):
    global allData
    allData[_str] = data2save
    return True


# this function calculates the normalized cross-correlation.
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


### MAIN

for filename in os.listdir(filepath):
    if filename.split('.mp3')[0]+'.pkl' in os.listdir(filepath_pkl) or not filename.endswith(".mp3"):
        continue
    else:
        # try:
            # filename = "0551 tzlilei haud - aluma aluma.mp3"
            allData = {}
            print(filename)
            print("loading...")
            y, sr = librosa.load(filepath + filename)
            yMax = np.max(y)
            idxStart = (np.where(y[1:sr * 40] > (yMax / 100)))[0][0]
            idxEnd = len(y) - sr*40 + (np.where(y[-sr*40:] > (yMax / 100) ))[0][-1]
            y = y[idxStart:idxEnd]
            # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
            hop_length = 512

            # Separate harmonics and percussives into two waveforms
            print("hpss...")
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            # Beat track on the percussive signal
            print("beat_frames...")
            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

            print("onset...")  # onset can also be aggregated over median, default is mean
            onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

            print("norm_cc...")
            onset_cc = np.zeros(int(np.ceil(len(beat_frames)/4)))
            onset_cc_t = np.zeros(int(np.ceil(len(beat_frames)/4)))
            onset_cc_max_idx = np.zeros(int(np.ceil(len(beat_frames)/4))) # add to code
            cnt = 0
            for beatIdx in range(0, len(beat_frames)-8, 4):
                if cnt % 10 == 0:
                    print(cnt)
                frameStart = beat_frames[beatIdx]
                frameEnd = beat_frames[beatIdx + 4]
                onSet_subVec = onset_env[frameStart:frameEnd]
                res = norm_cc(onset_env, onSet_subVec, frameStart, beat_frames[beatIdx + 8])
                onset_cc[cnt] = np.max(res[1:])
                onset_cc_t[cnt] = frameStart*(hop_length/sr)
                cnt += 1

            # onset_cc = data['onset_cc']
            # onset_cc_t = data['onset_cc_t']
            onset_cc_avg = np.mean(onset_cc[np.where(onset_cc_t > 0)])
            onset_cc_std = np.std(onset_cc[np.where(onset_cc_t > 0)])

            save2data('onset_cc', onset_cc)
            save2data('onset_cc_t', onset_cc_t)
            save2data('onset_cc_avg', onset_cc_avg)
            save2data('onset_cc_std', onset_cc_std)
            save2data('file_id', filename)

            filename_pkl = filename.split('.mp3')[0]+'.pkl'
            with open(filepath_pkl + filename_pkl, 'wb') as f:
                pickle.dump(allData, f)
                f.close()

            plt.figure(1)
            plt.plot(onset_cc_t[np.where(onset_cc_t > 0)],onset_cc[np.where(onset_cc_t > 0)])
            plt.title(filename)

            filename_img = filename.split('.mp3')[0]+'.png'
            plt.savefig(filepath_img + filename_img)
            filename_img = filename.split('.mp3')[0]+'.eps'
            plt.savefig(filepath_img + filename_img)

            plt.close()
            #plt.show()
        #except:
        #    continue
