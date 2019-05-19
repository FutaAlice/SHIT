#!/usr/bin/env python
# -*- coding: utf-8 -*-
import librosa
import crepe


def extract(y, sr, minfqr=120.0, maxfqr=720.0, step_size=40):
    times, freqs, confidence, activation = crepe.predict(y, sr, model_capacity='tiny',
                                                         viterbi=False, center=False, step_size=step_size)

    mute = confidence < 0.5
    freqs[freqs < minfqr] = 0
    freqs[freqs > maxfqr] = 0
    freqs[mute] = 0

    return freqs, times
