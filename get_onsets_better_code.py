import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gc


filepath = 'data/mp3/'
filepath_pkl = 'data/pkl/'
filepath_img = 'data/png/'


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



def segment_calc(segment, onset_envelope, beat_frames):

    # this function takes an onset segment size in beats and outputs C-C values
    # segment given in beats
    # onset_envelope is an onset strength time series (vector) extracted with lirbosa.onset.onset_strength()
    # beat_frames is the beats time series (vector) extracted with librosa.beat.beat_track()
    # hop length should be set to 512

    print("calculating normalized C-C for beat segment = " + str(segment))
    onscc = np.zeros(int(np.ceil(len(beat_frames)/segment)))
    onscc_t = np.zeros(int(np.ceil(len(beat_frames)/segment)))
    onscc_lag = np.zeros(int(np.ceil(len(beat_frames)/segment)))
    cnt = 0

    for beatIdx in range(0, len(beat_frames)-(2*segment), segment):
        if cnt%10 == 0:
            print(cnt)
        frame_start = beat_frames[beatIdx]
        frame_end = beat_frames[beatIdx+segment]
        onset_subvec = onset_envelope[frame_start:frame_end]
        result = norm_cc(onset_envelope, onset_subvec, frame_start, beat_frames[beatIdx+2*segment]+10)
        onscc[cnt] = np.max(result[1:])                     # this is C-C max value
        onscc_t[cnt] = frame_start * (hop_length / sr)    # this is the respective initial beat time for the X0 segment
        onscc_lag[cnt] = np.argmax(result[1:])       # this is the lag (in frames) between beat time and max value time
        cnt += 1

    onscc_lag = onscc_lag * (tempo / 60 / sr * hop_length)
    onscc_dict = {str(segment)+'_onscc_avg': np.nanmean(onscc[np.where(onscc_t > 0)]),
                  str(segment)+'_onscc_std': np.nanstd(onscc[np.where(onscc_t > 0)]),
                  str(segment)+'_onscc_lag_avg': np.nanmean(onscc_lag[np.where(onscc_t > 0)]),
                  str(segment)+'_onscc_lag_med': np.nanmedian(onscc_lag[np.where(onscc_t > 0)]),
                  str(segment)+'_onscc_lag_std': np.nanstd(onscc_lag[np.where(onscc_t > 0)])}

    return onscc, onscc_t, onscc_lag, onscc_dict


# MAIN

for filename in os.listdir(filepath):
    allData = {}
    print(filename)
    print("loading audio file...")
    y, sr = librosa.load(filepath + filename)   # , duration=60.0)
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
    tempo, track_beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)   # , bpm=120)
    gc.collect()
    print('beat times...')
    beat_times = librosa.frames_to_time(track_beat_frames, sr=sr)
    track_duration = len(y) / sr

    # onsets extraction from the percussive signal
    print("onsets extraction...")  # onset can also be aggregated over median, default is mean
    oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)

    onscc_bar, onscc_bar_t, onscc_bar_lag, bar_onscc_dict = segment_calc(4, oenv, track_beat_frames)
    onscc_half, onscc_half_t, onscc_half_lag, half_onscc_dict = segment_calc(2, oenv, track_beat_frames)
    onscc_quart, onscc_quart_t, onscc_quart_lag, quart_onscc_dict = segment_calc(1, oenv, track_beat_frames)
    dictlist = [bar_onscc_dict, half_onscc_dict, quart_onscc_dict]

    print('pickling...')
    save2data('file_id', filename)
    save2data('tempo', tempo)
    save2data('duration', track_duration)

    for i in dictlist:
        for key, value in i.items():
            save2data(key, value)

    save2data('onscc_bar', onscc_bar)
    save2data('onscc_bar_t', onscc_bar_t)
    save2data('onscc_bar_lag', onscc_bar_lag)
    save2data('onscc_half', onscc_half)
    save2data('onscc_half_t', onscc_half_t)
    save2data('onscc_half_lag', onscc_half_lag)
    save2data('onscc_quart', onscc_quart)
    save2data('onscc_quart_t', onscc_quart_t)
    save2data('onscc_quart_lag', onscc_quart_lag)

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

    plt.close()
