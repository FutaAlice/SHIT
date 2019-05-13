#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import melodia
import synthesize_wave
import synthesize_midi


def example_audio_file():
    # return librosa.util.example_audio_file()
    return './example/hongdou.wav'


def run(plot=True):
    filename = example_audio_file()

    print('loading audio file...')
    y, sr = librosa.load(filename, sr=44100, mono=True)

    print('extracting f0...')
    melody_raw, timestamps = melodia.get_raw_melody(y, sr)

    if plot:
        # plot melody
        plt.figure(figsize=(12, 12))

        # plot raw melody
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, melody_raw)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        melody_pos = melodia.raw_to_pos(melody_raw)
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, melody_pos)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        melody_abs = melodia.raw_to_abs(melody_raw)
        melody_abs[melody_abs <= 0] = None
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, melody_abs)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.show()

    # Ensure save directory exist.
    debug_dir = './debug/'
    pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)

    csv_path = debug_dir + 'melody.csv'
    wav_path = debug_dir + 'synth.wav'
    mid_path = debug_dir + 'synth.mid'

    # Save melody to csv file.
    print('saving csv melody file...')
    time_freq_matrix = np.array([timestamps, melody_raw]).T
    np.savetxt(csv_path, time_freq_matrix, delimiter=',')

    # Read csv file and synth wave.
    print('synthesizing wave...')
    data = np.loadtxt(csv_path, 'float', '#', ',')
    data = data.T
    assert data.shape[0] == 2
    freqs, times = data[1], data[0]
    freqs = melodia.raw_to_abs(freqs)
    synthesize_wave.melosynth(freqs, times, wav_path)

    # Read csv file and synth midi.
    print('synthesizing midi...')
    data = np.loadtxt(csv_path, 'float', '#', ',')
    data = data.T
    assert data.shape[0] == 2
    freqs = data[1]
    synthesize_midi.melosynth(freqs, mid_path, 60)


if __name__ == '__main__':
    run()
