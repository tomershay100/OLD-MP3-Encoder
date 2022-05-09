import numpy as np
from scipy import signal

import tables


class MP3Encoder:
    def __init__(self, wav_file):
        self.__wav_file = wav_file

    def encode(self):
        baseband_filter = self.__prototype_filter().astype('float32')

        subband_samples = np.zeros((self.__wav_file.get_num_of_ch(), tables.N_SUBBANDS, tables.FRAMES_PER_BLOCK),
                                   dtype='float32')

        # Main loop, executing until all samples have been processed.
        while self.__wav_file.get_num_of_processed_samples() < self.__wav_file.get_num_of_samples():

            # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
            for frm in range(tables.FRAMES_PER_BLOCK):
                samples_read = self.__wav_file.read_samples(tables.SHIFT_SIZE)

                # If all samples have been read, perform zero padding.
                if samples_read < tables.SHIFT_SIZE:
                    for ch in range(self.__wav_file.get_num_of_ch()):
                        self.__wav_file.audio[ch].insert(np.zeros(tables.SHIFT_SIZE - samples_read))

                # Filtering = dot product with reversed buffer.
                for ch in range(self.__wav_file.get_num_of_ch()):
                    subband_samples[ch, :, frm] = self.__subband_filtering(self.__wav_file.audio[ch].reversed(),
                                                                           baseband_filter)

            # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
            # Number of bits allocated in subband is either 0 or in range [2,15].
            scfindices = np.zeros((self.__wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='uint8')
            subband_bit_allocation = np.zeros((self.__wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='uint8')
            smr = np.zeros((self.__wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='float32')

            # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although
            # scaling is done later, its result is necessary for the psychoacoustic model and calculation of
            # sound pressure levels.
            for ch in range(self.__wav_file.get_num_of_ch()):
                # scfindices[ch, :] = get_scalefactors(subband_samples[ch, :, :], params.table.scalefactor) TODO
                # subband_bit_allocation[ch, :] = psycho.model1(wav_file.audio[ch].ordered(), params, scfindices) TODO
                pass

            subband_samples_quantized = np.zeros(subband_samples.shape, dtype='uint32')
            for ch in range(self.__wav_file.get_num_of_ch()):
                for sb in range(tables.N_SUBBANDS):
                    QCa = self.__wav_file.get_table().qca[subband_bit_allocation[ch, sb] - 2]
                    QCb = self.__wav_file.get_table().qcb[subband_bit_allocation[ch, sb] - 2]
                    scf = self.__wav_file.get_table().scalefactor[scfindices[ch, sb]]
                    ba = subband_bit_allocation[ch, sb]
                    for ind in range(tables.FRAMES_PER_BLOCK):
                        # subband_samples_quantized[ch, sb, ind] = quantization.quantization(subband_samples[ch, sb, ind], scf, ba, QCa, QCb) TODO
                        pass
            # Forming output bitsream and appending it to the output file.
            # bitstream_formatting(outmp3file, params, subband_bit_allocation, scfindices, subband_samples_quantized) TODO

    # Computes the prototype filter used in subband coding. The filter is a 512-point lowpass FIR h[n] with bandwidth
    # pi/64 and stopband starting at pi/32
    @staticmethod
    def __prototype_filter():
        lowpass_points = 512  # number of lowpass points
        sample_rate = np.pi  # setting sampling frequency
        pass_frequency = sample_rate / 128  # pass frequency
        stop_frequency = sample_rate / 32  # stop frequency

        filter = signal.remez(numtaps=lowpass_points, bands=[0, pass_frequency, stop_frequency, sample_rate],
                              desired=[2, 0], fs=2 * sample_rate)  # filter

        return filter

    # implementing the efficient version of the subband filter as specified by the MP3 standard
    #  x:  a new 512-point data buffer, in time-reversed order [x[n],x[n-1],...,x[n-511]].
    #  h:  The prototype filter of the filter bank
    # Returns: 32 new output samples
    @staticmethod
    def __subband_filtering(x, h):
        r = np.multiply(x, h)
        q = np.arange(64)
        c = np.sum((-1) ** np.arange(8)[:, np.newaxis] * r[q + 64 * np.arange(8)[:, np.newaxis]], axis=0)
        s = np.sum(np.cos(np.pi / 64. * (2 * np.arange(32)[:, np.newaxis] + 1) * (np.arange(q.shape[0]) - 16)) * c,
                   axis=1)
        return s
