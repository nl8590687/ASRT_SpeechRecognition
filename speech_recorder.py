#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

import pyaudio
import wave


def record_wave(wavfile,
                duration=10,
                channels=1,
                sampling_rate=16000,
                sampling_bits=16,
                chunk_size=1024,
                keyboard_interrupt='keep_audio'):
    """Record audio using the default audio device by PyAudio and Wave"""

    format_ = None
    if sampling_bits == 8:
        format_ = pyaudio.paInt8
    if sampling_bits == 16:
        format_ = pyaudio.paInt16
    elif sampling_bits == 24:
        format_ = pyaudio.paInt24
    elif sampling_bits == 32:
        format_ = pyaudio.paFloat32
    else:
        raise ValueError('Unsupported sampling bits')

    p = pyaudio.PyAudio()
    stream = p.open(format=format_,
                    channels=channels,
                    rate=sampling_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    frames = []

    print('Start to record with {}-seconds audio\n'
          'Type Ctrl-C to get an early stop (a shorter audio)'
          .format(duration))
    try:
        for _ in range(0, int(sampling_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
            print('.', end='', flush=True)
    except KeyboardInterrupt:
        if keyboard_interrupt == 'keep_audio':
            used_seconds = int(len(frames) * chunk_size / sampling_rate)
            print('\n-*- Early stop with {} seconds'.format(used_seconds))
        else:
            raise
    print('\nRecording finished')

    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Convert PCM frames to WAV... ', end='')
    wf = wave.open(wavfile, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format_))
    wf.setframerate(sampling_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print('OK')


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Simple Wave Audio Recorder',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--duration', type=int,
                        default=10, help='maximum duration in seconds')
    parser.add_argument('-r', '--sampling-rate', type=int,
                        default=16000, help='sampling rate')
    parser.add_argument('-b', '--sampling-bits', type=int,
                        default=16, choices=(8, 16, 24, 32), help='sampling bits')
    parser.add_argument('-c', '--channels', type=int,
                        default=1, help='audio channels')
    parser.add_argument('output', nargs='?', default='output.wav', help='audio file to store audio stream')
    args = parser.parse_args()
    record_wave(args.output, duration=args.duration,
                channels=args.channels,
                sampling_bits=args.sampling_bits,
                sampling_rate=args.sampling_rate)
