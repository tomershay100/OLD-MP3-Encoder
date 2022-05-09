# Read the WAV into a buffer. Only standard PCM WAVE supported.
import os
import struct
import sys

import numpy as np

FRAME_SIZE = 512
FFT_SIZE = 512
N_SUBBANDS = 32
SHIFT_SIZE = 32
SLOT_SIZE = 32
FRAMES_PER_BLOCK = 12

EPS = 1e-6
INF = 123456


class WAVFile:
    # sample_rate, num_of_ch, bits_per_sample used for PCM
    def __init__(self, file_path, sample_rate=0, num_of_ch=0, bits_per_sample=0):
        self.__file_path = file_path
        self.__file = open(self.__file_path, 'rb')

        if self.__file_path[-3:] == 'wav':
            self.__read_header()
        elif self.__file_path[-3:] == 'pcm':
            if sample_rate == 0 or num_of_ch == 0 or bits_per_sample == 0:
                sys.exit('Please provide sampling frequency, number of channels \
                      and number of bits per sample for PCM audio file.')
            self.__sample_rate = sample_rate
            self.__num_of_ch = num_of_ch
            self.__bits_per_sample = bits_per_sample
            self.__num_of_samples = os.path.getsize(self.__file_path) * 8 / self.__bits_per_sample / self.__num_of_ch

        if self.__bits_per_sample == 8:
            self.__datatype = 'int8'
        elif self.__bits_per_sample == 16:
            self.__datatype = 'int16'
        else:
            self.__datatype = 'int32'
        self.__num_of_processed_samples = 0
        self.__audio = []
        for ch in range(self.__num_of_ch):
            self.__audio.append(CircBuffer(FRAME_SIZE))

    def get_params(self):
        pass

    # Reads the header information to check if it is a valid file with PCM audio samples.
    def __read_header(self):
        buffer = self.__file.read(128)

        idx = buffer.find(b'RIFF')  # bytes 1 - 4
        if idx == -1:
            sys.exit('Bad WAVE file.')
        idx += 4
        self.__chunk_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 5 - 8

        idx = buffer.find(b'WAVE')  # bytes 9 - 12
        if idx == -1:
            sys.exit('Bad WAVE file.')
        idx = buffer.find(b'fmt ')  # bytes 13 - 16 (format chunk marker)
        if idx == -1:
            sys.exit('Bad WAVE file.')

        idx += 4
        sub_chunk1_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 17 - 20, size of fmt section
        if sub_chunk1_size != 16:
            sys.exit('Unsupported WAVE file, compression used instead of PCM.')

        idx += 4
        format_type = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 21 - 22
        if format_type != 1:  # 1 for PCM
            sys.exit('Unsupported WAVE file, compression used instead of PCM.')

        idx += 2
        self.__num_of_ch = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 23 - 24

        idx += 2
        self.__sample_rate = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 25 - 28

        idx += 4
        self.__byte_rate = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 29 - 32
        # ByteRate = (SampleRate * BitsPerSample * Channels) / 8

        idx += 4
        self.__block_align = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 33 - 34
        # BlockAlign = BitsPerSample * Channels / 8

        idx += 2
        self.__bits_per_sample = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 35 - 36
        if not (self.__bits_per_sample in (8, 16, 32)):
            sys.exit('Unsupported WAVE file, samples not int8, int16 or int32 type.')

        idx = buffer.find(b'data')  # bytes 37 - 40
        if idx == -1:
            sys.exit('Bad WAVE file.')

        idx += 4
        sub_chunk2_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 41 - 44, size of data section
        self.__num_of_samples = sub_chunk2_size * 8 / self.__bits_per_sample / self.__num_of_ch

        self.__file.seek(idx + 4)


# Circular buffer used for audio input.
class CircBuffer:
    def __init__(self, size, datatype='float32'):
        self.size = size
        self.pos = 0
        self.samples = np.zeros(size, dtype=datatype)
