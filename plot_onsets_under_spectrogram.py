import librosa.display
import numpy as np
# from matplotlib import __version__  as matplotlib__version__
import matplotlib.pyplot as plt
import pickle
import os

# np.seterr(all='raise')
filepath = 'beats/mp3/'
filepath_pkl = 'beats/pkl/'
filepath_img = 'beats/img/'
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


# MAIN

for filename in os.listdir(filepath):
    if not filename.endswith(".mp3"): # filename.split('.mp3')[0]+'.pkl' in os.listdir(filepath_pkl) or
        continue
    else:
        # try:
            # filename = "0551 tzlilei haud - aluma aluma.mp3"
            allData = {}
            print(filename)
            print("loading...")
            y, sr = librosa.load(filepath + filename, duration=60.0)
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
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            print("onset...")  # onset can also be aggregated over median, default is mean
            oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

            print('stft and times...')
            D = np.abs(librosa.stft(y_percussive))
            times = librosa.frames_to_time(np.arange(D.shape[1]))

            print("norm_cc for half bar")
            seg = 2  # set desired length of segment to test
            onscc_half = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_half_t = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_half_lag = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            cnt = 0

            for beatIdx in range(0, len(beat_frames)-2*seg, seg):
                if cnt % 10 == 0:
                    print(cnt)
                frameStart = beat_frames[beatIdx]
                frameEnd = beat_frames[beatIdx + seg]
                onSet_subVec = oenv[frameStart:frameEnd]
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx + 2*seg])
                onscc_half[cnt] = np.max(res[1:])
                onscc_half_t[cnt] = frameStart*(hop_length/sr)
                onscc_half_lag[cnt] = np.argmax(res[1:])
                cnt += 1

            # onscc_half = data['onscc_half']
            # onscc_half_t = data['onscc_half_t']
            onscc_half_avg = np.mean(onscc_half[np.where(onscc_half_t > 0)])
            onscc_half_std = np.std(onscc_half[np.where(onscc_half_t > 0)])
            onscc_half_lag = onscc_half_lag * (tempo / 60 / sr * hop_length)
            onscc_half_lag_avg = np.mean(onscc_half_lag[np.where(onscc_half_t > 0)])
            onscc_half_lag_med = np.median(onscc_half_lag[np.where(onscc_half_t > 0)])
            onscc_half_lag_std = np.std(onscc_half_lag[np.where(onscc_half_t > 0)])

            print("norm_cc for full bar")
            seg = 4  # set desired length of segment to test
            onscc_bar = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_bar_t = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_bar_lag = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            cnt = 0

            for beatIdx in range(0, len(beat_frames)-2*seg, seg):
                if cnt%10 == 0:
                    print(cnt)
                frameStart = beat_frames[beatIdx]
                frameEnd = beat_frames[beatIdx+seg]
                onSet_subVec = oenv[frameStart:frameEnd]
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx+2*seg])
                onscc_bar[cnt] = np.max(res[1:])
                onscc_bar_t[cnt] = frameStart*(hop_length/sr)
                onscc_bar_lag[cnt] = np.argmax(res[1:])
                cnt += 1

            # onscc_bar = data['onscc_bar']
            # onscc_bar_t = data['onscc_bar_t']
            onscc_bar_avg = np.mean(onscc_bar[np.where(onscc_bar_t > 0)])
            onscc_bar_std = np.std(onscc_bar[np.where(onscc_bar_t > 0)])
            onscc_bar_lag = onscc_bar_lag*(tempo/60/sr*hop_length)
            onscc_bar_lag_avg = np.mean(onscc_bar_lag[np.where(onscc_bar_t > 0)])
            onscc_bar_lag_med = np.median(onscc_bar_lag[np.where(onscc_bar_t > 0)])
            onscc_bar_lag_std = np.std(onscc_bar_lag[np.where(onscc_bar_t > 0)])

            print("norm_cc for quarter")
            seg = 1  # set desired length of segment (in beats) to test
            onscc_quart = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_quart_t = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_quart_lag = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            cnt = 0

            for beatIdx in range(0, len(beat_frames)-2*seg, seg):
                if cnt%10 == 0:
                    print(cnt)
                frameStart = beat_frames[beatIdx]
                frameEnd = beat_frames[beatIdx+seg]
                onSet_subVec = oenv[frameStart:frameEnd]
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx+2*seg])
                onscc_quart[cnt] = np.max(res[1:])
                onscc_quart_t[cnt] = frameStart*(hop_length/sr)
                onscc_quart_lag[cnt] = np.argmax(res[1:])
                cnt += 1

            # onscc_quart = data['onscc_quart']
            # onscc_quart_t = data['onscc_quart_t']
            onscc_quart_avg = np.mean(onscc_quart[np.where(onscc_quart_t > 0)])
            onscc_quart_std = np.std(onscc_quart[np.where(onscc_quart_t > 0)])
            onscc_quart_lag = onscc_quart_lag*(tempo/60/sr*hop_length)
            onscc_quart_lag_avg = np.mean(onscc_quart_lag[np.where(onscc_quart_t > 0)])
            onscc_quart_lag_med = np.median(onscc_quart_lag[np.where(onscc_quart_t > 0)])
            onscc_quart_lag_std = np.std(onscc_quart_lag[np.where(onscc_quart_t > 0)])

            save2data('file_id', filename)
            save2data('onscc_half', onscc_half)
            save2data('onscc_half_t', onscc_half_t)
            save2data('onscc_half_avg', onscc_half_avg)
            save2data('onscc_half_std', onscc_half_std)
            save2data('onscc_half_lag_avg', onscc_half_lag_avg)
            save2data('onscc_half_lag_med', onscc_half_lag_med)
            save2data('onscc_half_lag_std', onscc_half_lag_std)
            save2data('onscc_bar', onscc_bar)
            save2data('onscc_bar_t', onscc_bar_t)
            save2data('onscc_bar_avg', onscc_bar_avg)
            save2data('onscc_bar_std', onscc_bar_std)
            save2data('onscc_bar_lag_avg', onscc_bar_lag_avg)
            save2data('onscc_bar_lag_med', onscc_bar_lag_med)
            save2data('onscc_bar_lag_std', onscc_bar_lag_std)
            save2data('onscc_quart', onscc_quart)
            save2data('onscc_quart_t', onscc_quart_t)
            save2data('onscc_quart_avg', onscc_quart_avg)
            save2data('onscc_quart_std', onscc_quart_std)
            save2data('onscc_quart_lag_avg', onscc_quart_lag_avg)
            save2data('onscc_quart_lag_med', onscc_quart_lag_med)
            save2data('onscc_quart_lag_std', onscc_quart_lag_std)

            filename_pkl = filename.split('.mp3')[0]+'.pkl'
            with open(filepath_pkl + filename_pkl, 'wb') as f:
                pickle.dump(allData, f)
                f.close()

            print('plotting...')
            plt.figure(figsize=(10, 4))
            librosa.display.waveplot(y_harmonic, sr=sr, alpha=0.2)
            librosa.display.waveplot(y_percussive, sr=sr, color='r', alpha=0.3)

            ax1 = plt.subplot(211)
            librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
            # ax1.set_xlim(left=70, right=90)
            plt.xlabel('time (s)')
            plt.title(filename + ' --- Power Spectrogram')

            ax2 = plt.subplot(212)
            plt.margins(x=0.001, y=0.005)
            plt.plot(onscc_half_t[np.where(onscc_half_t > 0)], onscc_half[np.where(onscc_half_t > 0)],
                     label='onsets CC, half bar segment', zorder=2)
            plt.plot(onscc_bar_t[np.where(onscc_bar_t > 0)], onscc_bar[np.where(onscc_bar_t > 0)],
                     label='onsets CC, bar segment', zorder=1)
            plt.plot(onscc_quart_t[np.where(onscc_quart_t > 0)], onscc_quart[np.where(onscc_quart_t > 0)],
                     label='onsets CC, quarter segment', zorder=1)
            bottom, top = plt.ylim()
            plt.vlines(beat_times[0::4], bottom, 1, linestyle='dotted', label='bars', zorder=0, alpha=0.4)
            plt.title('Onsets cross-correlation')
            plt.ylabel('normalized onsets CC')
            plt.legend(loc='center', fancybox=True, ncol=4, bbox_to_anchor=(0.5, -0.5))
            plt.tight_layout()
            # plt.show()
            filename_img = filename.split('.mp3')[0]+'.png'
            plt.savefig(filepath_img + filename_img)
            plt.close()
            #plt.show()
        #except:
        #    continue
