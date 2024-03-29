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
get_onsets.py is currently the main code file. Place the audio files (mp3 format) you 
wish to examine in the 'data/mp3/' folder, and run the code. For each audio file, a 
data output pickle files is created in the 'data/pkl/' folder, and the onsets-CC graphs 
are created in the 'data/img/' folder. To get all of the pickled data from multiple
files into one csv file, use merge_pickles.py. If you wish to get a graph that 
includes the audio power spectrogram, you can use plot_onsets_under_spectrogram.py.

## Authors
Adam Yodfat, PhD candidate, Musicology dept., Hebrew University of Jerusalem (This project is part of a larger academic research about Israeli Popular Music)

Amir Avni

## Citation

For the full dissertation (in Hebrew): https://www.academia.edu/45342000/אלף_שירים_ושיר_חמישה_עשורים_של_שירי_מזרחית_ורוק_בישראל_איפיון_מוסיקלי
For Abstract and contents in English: https://www.academia.edu/45342052/A_Thousand_Songs_and_a_Song_Five_Decades_of_Mizrahit_and_Rock_Songs_in_Israel_Musical_Analysis_English_Abstract_and_Contents_

To cite this work, please refer to the dissertation: Adam Yodfat, "A Thousand Songs and a Song: Five Decades of Mizrahit and Rock Songs in Israel - Musical Analysis", PhD Dissertation (The Hebrew University of Jerusalem, 2020).

We also intend to publish this work as a stand alone paper in English in the future.

