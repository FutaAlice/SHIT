#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Melody Extraction in Python with Melodia
http://www.justinsalamon.com/news/melody-extraction-in-python-with-melodia
"""

import numpy as np
import vamp


def get_raw_melody(y, sr, minfqr=55.0, maxfqr=1760.0, voicing=0.2, minpeaksalience=0.0):
    """ Use 'Melodia' to extract the pitch contour of a melody from a song.

    :param y: input signal, np.ndarray [shape=(n,)]
    :param sr: sample rate
    :param minfqr: minimum frequency in Hertz (default 55.0)
    :param maxfqr: maximum frequency in Hertz (default 1760.0)
    :param voicing: voicing tolerance. Greater values will result in more pitch contours included in the final melody.
        Smaller values will result in less pitch contours included in the final melody (default 0.2).
    :param minpeaksalience: is a hack to avoid silence turning into junk contours when analyzing monophonic recordings
        (e.g. solo voice with no accompaniment). Generally you want to leave this untouched (default 0.0).

    :return: a tuple of melody and timestamp, unvoiced (=no melody) sections as negative values.
    """
    # Parameter values are specified by providing a dicionary to the optional "parameters" parameter:
    params = {"minfqr": minfqr, "maxfqr": maxfqr, "voicing": voicing, "minpeaksalience": minpeaksalience}

    # Exracting the melody using Melodia(or 'melodiaviz' for collect matrix).
    data = vamp.collect(y, sr, "mtg-melodia:melodia", parameters=params)

    # Hop size is *always* equal to 128/44100.0 = 2.9 ms
    # Melody, an array of pitch values.
    (hop, melody) = data['vector']

    # SUPER IMPORTANT SUPER IMPORTANT
    # For reasons internal to the vamp architecture, THE TIMESTAMP OF THE FIRST VALUE IN THE MELODY ARRAY IS ALWAYS:
    first_timestamp = 8 * hop.to_float()

    # This means that the timestamp of the pitch value at index i (starting with i=0) is given by:
    timestamps = first_timestamp + np.arange(len(melody)) * hop.to_float()

    return melody, timestamps


def raw_to_pos(melody):
    """
    A clearer option is to get rid of the negative values before plotting
    """
    melody_pos = np.copy(melody[:])
    melody_pos[melody <= 0] = None
    return melody_pos


def raw_to_abs(melody):
    """
    TODO
    """
    melody_abs = np.abs(melody)
    melody_abs[melody == -55] = 0
    melody_abs[melody == -110] = 0
    melody_abs[melody == -220] = 0
    melody_abs[melody == -440] = 0
    return melody_abs


def raw_to_cents(melody):
    """
    You might want to plot the pitch sequence in cents rather than in Hz.
    This especially makes sense if you are comparing two or more pitch sequences to each other
    e.g. comparing an estimate against a reference.
    """
    melody_cents = 1200 * np.log2(np.abs(melody) / 55.0)
    melody_cents[melody == -220] = None
    melody_cents[melody == -440] = None
    return melody_cents

