# Circular buffer used for audio input.
import numpy as np

NUM_OF_SUBBANDS = 32
FRAMES_PER_BLOCK = 12


class CircBuffer:
    def __init__(self, size, datatype='float32'):
        self.__size = size
        self.__pos = 0
        self.__buffer = np.zeros(size, dtype=datatype)

    def insert(self, frame):
        length = len(frame)
        if self.__pos + length <= self.__size:
            self.__buffer[self.__pos:self.__pos + length] = frame
        else:
            overhead = length - (self.__size - self.__pos)
            self.__buffer[self.__pos:self.__size] = frame[:-overhead]
            self.__buffer[0:overhead] = frame[-overhead:]
        self.__pos += length
        self.__pos %= self.__size

    def ordered(self):
        return np.concatenate((self.__buffer[self.__pos:], self.__buffer[:self.__pos]))

    def reversed(self):
        return np.concatenate((self.__buffer[self.__pos - 1::-1], self.__buffer[:self.__pos - 1:-1]))


# Form an array of bytes and fill it as a bitstream
class BitStream:
    # Initialize Output Buffer with size in bytes
    def __init__(self, size):
        self.__size = size
        self.__pos = 0
        self.__data = np.zeros(size, dtype='uint8')

    # Insert lowest nbits of data in Output Buffer
    def insert(self, data, nbits, invert_msb=False):
        if invert_msb:
            data = self.__invert_msb(data, nbits)
        data_in_bytes = self.__split_in_bytes(data, nbits, self.__pos & 0x7)
        ind = self.__pos // 8
        for byte in data_in_bytes:
            if ind >= self.__size:
                break
            self.__data[ind] |= byte
            ind += 1
        self.__pos += nbits

    # Set all bits higher than nbits to zero.
    @staticmethod
    def __mask_upper_bits(data, nbits):
        mask = ~((0xFFFFFFFF << nbits) & 0xFFFFFFFF)
        return data & mask

    # Invert MSB of data, data being only lowest nbits.
    @staticmethod
    def __invert_msb(data, nbits):
        mask = 1 << (nbits - 1)
        return data ^ mask

    # Split input data in bytes to allow insertion in buffer by OR operation.
    def __split_in_bytes(self, data, nbits, pos):
        data = self.__mask_upper_bits(data, nbits)
        shift = (8 - (nbits & 0x7) + 8 - pos) & 0x7
        data <<= shift
        nbits += shift
        data_in_bytes = ()
        loop_count = 1 + (nbits - 1) // 8
        for i in range(loop_count):
            data_in_bytes = (data & 0xFF,) + data_in_bytes
            data >>= 8
        return data_in_bytes

    @property
    def data(self):
        return self.__data


# Calculate scale factors for subbands. Scale factor is equal to the smallest number in the table
# greater than all the subband samples in a particular subband. Scalefactor table indices are returned.
def get_scale_factors(subband_samples, scale_factor_table):
    scale_factor_indexes = np.zeros(subband_samples.shape[0:-1], dtype='uint8')
    subbands_max_vals = np.max(np.absolute(subband_samples), axis=1)
    for subband in range(NUM_OF_SUBBANDS):
        i = 0
        while scale_factor_table[i + 1] > subbands_max_vals[subband]:
            i += 1
        scale_factor_indexes[subband] = i
    return scale_factor_indexes


# Form a MPEG-1 Layer 1 bitstream and append it to output file.
def bitstream_formatting(wav_file, alloc, scale_factor, sample):
    buffer = BitStream((wav_file.num_of_slots + wav_file.padbit) * 4)

    buffer.insert(wav_file.header, 32)
    wav_file.updateheader()

    for subband in range(NUM_OF_SUBBANDS):
        for channel in range(wav_file.num_of_ch):
            buffer.insert(np.max((alloc[channel][subband] - 1, 0)), 4)

    for subband in range(NUM_OF_SUBBANDS):
        for channel in range(wav_file.num_of_ch):
            if alloc[channel][subband] != 0:
                buffer.insert(scale_factor[channel][subband], 6)

    for s in range(FRAMES_PER_BLOCK):
        for subband in range(NUM_OF_SUBBANDS):
            for channel in range(wav_file.num_of_ch):
                if alloc[channel][subband] != 0:
                    buffer.insert(sample[channel][subband][s], alloc[channel][subband], True)

    fp = open(wav_file.file_path, 'a+')
    buffer.data.tofile(fp)
    fp.close()
