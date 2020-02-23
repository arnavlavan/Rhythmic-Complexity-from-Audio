# Rhythmic Complexity from Audio Using Note Onsets Cross-Correlation
This code is part of a research project that proposes a new way to measure the rhythmic complexity of music. 
The method is based on audio feature analysis using the librosa package for Python.

## General Outline:
1. Audio is extracted and separated to harmonic and percussive componnents
2. Beats and note onset envelope are extracted from the percussive componnent
3. Three segment lengths are defined - full bar (4 beats), half bar (2 beats), and quarter bar (1 beat).
4. For each segment length, every segment in the song is compared using cross-correlation to its following same-length segment, and all the same-length segments that exist between the two. 
5. The maximal cross-correlation value is chosen to represnt the starting beat of each segment, and thus a cross-correlation vector is computed for the entire song.
6. Since the process is repeated for each segment length, every audio segment yields 3 cross-correlation vectors. 
7. High cross-correlation values are suggested to represent a low level of rhythmic complexity, and vice versa - low cross-correlation values are suggested to represent a high level of rhythmic complexity.

## Prerequisites
Python (version 3.6+),
Python packages: librosa, numpy, matplotlib

## Getting Started
get_onsets.py is currently the main code file. Place the audio files (mp3 format) you wish to examine in the 'data/mp3/' folder, and run the code. For each audio file, a data output pickle files is created in the 'data/pkl/' folder, and the onsets-CC graphs are created in the 'data/img/' folder.

## Authors
Adam Yodfat, PhD candidate, Musicology dept., Hebrew University of Jerusalem (This project is part of a larger academic research about Israeli Popular Music)
Amir Avni