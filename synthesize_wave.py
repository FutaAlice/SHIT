#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import wave


def wavwrite(x, filename, fs=44100, n=16):
    """ Synthesize signal x into a wavefile on disk. The values of x must be in the range [-1,1].

    :param x: Signal to synthesize, numpy.array
    :param filename: Path of output wavfile.
    :param fs: Sampling frequency, by default 44100.
    :param n: Bit depth, by default 16.
    """
    max_vol = 2**15-1.0  # maximum amplitude
    x = x * max_vol  # scale x
    # convert x to string format expected by wave
    signal = b"".join((wave.struct.pack('h', int(item)) for item in x))
    wv = wave.open(filename, 'w')
    nchannels = 1
    sampwidth = int(n / 8)  # in bytes
    framerate = fs
    nframe = 0  # no limit
    comptype = 'NONE'
    compname = 'not compressed'
    wv.setparams((nchannels, sampwidth, framerate, nframe, comptype, compname))
    wv.writeframes(signal)
    wv.close()


def melosynth(freqs, times, outputfile, fs=16000, n_harmonics=1, square=False, useneg=False):
    # Preprocess input parameters
    fs = int(float(fs))
    n_harmonics = int(n_harmonics)

    # Impute silence if start time > 0
    if times[0] > 0:
        estimated_hop = np.median(np.diff(times))
        prev_time = max(times[0] - estimated_hop, 0)
        times = np.insert(times, 0, prev_time)
        freqs = np.insert(freqs, 0, 0)

    # Generating wave...
    signal = []

    translen = 0.010  # duration (in seconds) for fade in/out and freq interp
    phase = np.zeros(n_harmonics)  # start phase for all harmonics
    f_prev = 0  # previous frequency
    t_prev = 0  # previous timestamp
    for t, f in zip(times, freqs):

        # Compute number of samples to synthesize
        nsamples = int(np.round((t - t_prev) * fs))

        if nsamples > 0:
            # calculate transition length (in samples)
            translen_sm = float(min(np.round(translen*fs), nsamples))

            # Generate frequency series
            freq_series = np.ones(nsamples) * f_prev

            # Interpolate between non-zero frequencies
            if f_prev > 0 and f > 0:
                freq_series += np.minimum(np.arange(nsamples)/translen_sm, 1) *\
                               (f - f_prev)
            elif f > 0:
                freq_series = np.ones(nsamples) * f

            # Repeat for each harmonic
            samples = np.zeros(nsamples)
            for h in range(n_harmonics):
                # Determine harmonic num (h+1 for sawtooth, 2h+1 for square)
                h_num = 2*h+1 if square else h+1
                # Compute the phase of each sample
                phasors = 2 * np.pi * h_num * freq_series / float(fs)
                phases = phase[h] + np.cumsum(phasors)
                # Compute sample values and add
                samples += np.sin(phases) / h_num
                # Update phase
                phase[h] = phases[-1]

            # Fade in/out and silence
            if f_prev == 0 and f > 0:
                samples *= np.minimum(np.arange(nsamples)/translen_sm, 1)
            if f_prev > 0 and f == 0:
                samples *= np.maximum(1 - (np.arange(nsamples)/translen_sm), 0)
            if f_prev == 0 and f == 0:
                samples *= 0

            # Append samples
            signal.extend(samples)

        t_prev = t
        f_prev = f

    # Normalize signal
    signal = np.asarray(signal)
    signal *= 0.8 / float(np.max(signal))

    # Saving wav file...
    wavwrite(np.asarray(signal), outputfile, fs)
