# Read the WAV into a buffer. Only standard PCM WAVE supported.
import os
import struct
import sys

import numpy as np

from tables import Tables

FRAME_SIZE = 512


class WAVFile:
    # sample_rate, num_of_ch, bits_per_sample used for PCM
    def __init__(self, file_path, bitrate=320, sample_rate=0, num_of_ch=0, bits_per_sample=0):
        self.__bitrate = bitrate
        if self.__bitrate == 32 and self.__num_of_ch == 2:
            sys.exit('Bitrate of 32Kbits/s is insufficient for encoding of stereo audio.')

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
        self.audio = []
        for ch in range(self.__num_of_ch):
            self.audio.append(CircBuffer(FRAME_SIZE))

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
        if self.__sample_rate not in (32000, 44100, 48000):
            sys.exit('Unsupported sampling frequency.')
        self.__sample_rate_code = {44100: 0b00, 48000: 0b01, 32000: 0b10}.get(self.__sample_rate)

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

        self.__num_of_slots = 12 * self.__bitrate * 1000 // self.__sample_rate
        self.__copyright = 0
        self.__original = 0
        self.__chmode = 0b11 if self.__num_of_ch == 1 else 0b10
        self.__modext = 0b10
        self.__sync_word = 0b11111111111
        self.__mpeg_version = 0b11
        self.__layer = 0b11
        self.__crc = 0b1
        self.__emphasis = 0b00
        self.__pad_bit = 0
        self.__rest = 0

        self.__header = (self.__sync_word << 21 | self.__mpeg_version << 19 |
                         self.__layer << 17 | self.__crc << 16 |
                         self.__bitrate << 7 | self.__sample_rate_code << 10 |
                         self.__pad_bit << 9 | self.__chmode << 6 |
                         self.__modext << 4 | self.__copyright << 3 |
                         self.__original << 2 | self.__emphasis)

        self.__table = Tables(self.__sample_rate, self.__bitrate)

    # Update pad_bit in header for current frame.
    def update_header(self):
        self.__need_padding()
        if self.__pad_bit:
            self.__header |= 0x00000200
        else:
            self.__header &= 0xFFFFFDFF

    # To ensure the constant bitrate, for fs=44100 padding is sometimes needed.
    def __need_padding(self):
        dif = (self.__bitrate * 1000 * 12) % self.__sample_rate
        self.__rest -= dif
        if self.__rest < 0:
            self.__rest += self.__sample_rate
            self.__pad_bit = 1
        else:
            self.__pad_bit = 0

    # Read number of samples from WAVE file and insert it in circular buffer.
    def read_samples(self, num_of_samples):
        read_size = self.__num_of_ch * num_of_samples
        frame = np.fromfile(self.__file, self.__datatype, read_size)
        frame.shape = (-1, self.__num_of_ch)
        for ch in range(self.__num_of_ch):
            self.audio[ch].insert(frame[:, ch].astype('float32') / (1 << self.__bits_per_sample - 1))
        self.__num_of_processed_samples += frame.shape[0]
        return frame.shape[0]

    def get_num_of_ch(self):
        return self.__num_of_ch

    def get_num_of_processed_samples(self):
        return self.__num_of_processed_samples

    def get_num_of_samples(self):
        return self.__num_of_samples

    def get_table(self):
        return self.__table


# Circular buffer used for audio input.
class CircBuffer:
    def __init__(self, size, datatype='float32'):
        self.size = size
        self.pos = 0
        self.samples = np.zeros(size, dtype=datatype)

    def insert(self, frame):
        length = len(frame)
        if self.pos + length <= self.size:
            self.samples[self.pos:self.pos + length] = frame
        else:
            overhead = length - (self.size - self.pos)
            self.samples[self.pos:self.size] = frame[:-overhead]
            self.samples[0:overhead] = frame[-overhead:]
        self.pos += length
        self.pos %= self.size

    def ordered(self):
        return np.concatenate((self.samples[self.pos:], self.samples[:self.pos]))

    def reversed(self):
        return np.concatenate((self.samples[self.pos - 1::-1], self.samples[:self.pos - 1:-1]))
