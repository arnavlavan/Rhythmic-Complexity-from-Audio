import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gc

filepath = 'C:/Users/OWNER/Dropbox/1phd/database/files/'
# filepath = 'data/mp3/'
filepath_pkl = 'data/pkl/'
filepath_img = 'data/img/'


def save2data(_str, data2save):
    global allData
    allData[_str] = data2save
    return True


# this function calculates the normalized cross-correlation
def norm_cc(a, b, start_idx, end_idx):
    res = np.zeros(end_idx + start_idx)
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
    if filename.split('.mp3')[0]+'.pkl' in os.listdir(filepath_pkl) or not filename.endswith(".mp3"):
        continue
    else:
        try:
            # filename = "0551 tzlilei haud - aluma aluma.mp3"
            allData = {}
            print(filename)
            print("loading audio file...")
            y, sr = librosa.load(filepath + filename)
            y, trim_index = librosa.effects.trim(y)      # trim leading and trailing silence in track
            hop_length = 512                             # set frame duration in samples

            # Separate harmonics and percussives into two waveforms
            print("harmonic/percussive separation...")
            D = librosa.stft(y)
            gc.collect()
            harm, perc = librosa.decompose.hpss(D)
            gc.collect()
            y_harmonic = librosa.istft(harm)
            gc.collect()
            y_percussive = librosa.istft(perc)
            gc.collect()

            # Beat track on the percussive signal
            print('beat tracking...')
            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr) # , bpm=120)
            gc.collect()
            print('beat times...')
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            track_duration = len(y) / sr

            # onsets extraction from the percussive signal
            print("onsets extraction...")  # onset can also be aggregated over median, default is mean
            oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

            print("calculating normalized C-C for half bar segment")
            seg = 2  # set desired beat length of segment to test
            onscc_half = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_half_t = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            onscc_half_lag = np.zeros(int(np.ceil(len(beat_frames)/seg)))
            cnt = 0

            for beatIdx in range(0, len(beat_frames)-(2*seg), seg):
                if cnt % 10 == 0:
                    print(cnt)
                frameStart = beat_frames[beatIdx]
                frameEnd = beat_frames[beatIdx + seg]
                onSet_subVec = oenv[frameStart:frameEnd]
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx + 2*seg]+10)
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

            print("calculating normalized C-C for bar segment")
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
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx+2*seg]+10)
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

            print("calculating normalized C-C for quarter segment")
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
                res = norm_cc(oenv, onSet_subVec, frameStart, beat_frames[beatIdx+2*seg]+10)
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

            print('pickling...')
            save2data('file_id', filename)
            save2data('tempo', tempo)
            save2data('duration', track_duration)
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
            librosa.display.waveplot(y_harmonic, sr=sr, alpha=0.2, label='harmonic waveform')
            librosa.display.waveplot(y_percussive, sr=sr, color='r', alpha=0.3, label='percussive waveform')
            plt.plot(onscc_bar_t[np.where(onscc_bar_t > 0)], onscc_bar[np.where(onscc_bar_t > 0)],
                     label='onsets C-C, bar segment', zorder=3, alpha=0.8, color='purple')
            plt.plot(onscc_half_t[np.where(onscc_half_t > 0)], onscc_half[np.where(onscc_half_t > 0)],
                     label='onsets C-C, half bar segment', zorder=2, alpha=0.8, color='green')
            plt.plot(onscc_quart_t[np.where(onscc_quart_t > 0)], onscc_quart[np.where(onscc_quart_t > 0)],
                     label='onsets C-C, quarter segment', zorder=1, alpha=0.5, color='b')
            plt.ylim(0, 1)
            plt.vlines(beat_times[0::4], 0, 1, linestyle='dotted', label='bars', zorder=0, alpha=0.4)
            plt.title(filename.split('.mp3')[0]+', inferred tempo: '+str(int(tempo))+' bpm')
            plt.xlabel('time (s)')
            plt.ylabel('normalized values')
            plt.legend(bbox_to_anchor=(0.5, 0.5), loc="center", fancybox=True, ncol=3)
            plt.tight_layout()
            # plt.show()
            filename_img = filename.split('.mp3')[0]+'.png'
            plt.savefig(filepath_img + filename_img)
            filename_img = filename.split('.mp3')[0]+'.eps'
            plt.savefig(filepath_img + filename_img)

            plt.close()

        except:
            continue
