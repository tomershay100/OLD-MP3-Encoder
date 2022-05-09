import os
import sys

import numpy as np

import tables
from WAV_File import WAVFile

if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit('Unexpected number of arguments.')
    if len(sys.argv) < 2:
        sys.exit('No directory specified.')
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        sys.exit('File not found.')

    wav_file = WAVFile(file_path)

    # baseband_filter = prototype_filter().astype('float32') TODO
    subband_samples = np.zeros((wav_file.get_num_of_ch(), tables.N_SUBBANDS, tables.FRAMES_PER_BLOCK), dtype='float32')

    # Main loop, executing until all samples have been processed.
    while wav_file.get_num_of_processed_samples() < wav_file.get_num_of_samples():

        # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
        for frm in range(tables.FRAMES_PER_BLOCK):
            samples_read = wav_file.read_samples(tables.SHIFT_SIZE)

            # If all samples have been read, perform zero padding.
            if samples_read < tables.SHIFT_SIZE:
                for ch in range(wav_file.get_num_of_ch()):
                    wav_file.audio[ch].insert(np.zeros(tables.SHIFT_SIZE - samples_read))

            # Filtering = dot product with reversed buffer.
            for ch in range(wav_file.get_num_of_ch()):
                # subband_samples[ch, :, frm] = subband_filtering.subband_filtering(wav_file.audio[ch].reversed(), baseband_filter) TODO
                pass

        # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
        # Number of bits allocated in subband is either 0 or in range [2,15].
        scfindices = np.zeros((wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='uint8')
        subband_bit_allocation = np.zeros((wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='uint8')
        smr = np.zeros((wav_file.get_num_of_ch(), tables.N_SUBBANDS), dtype='float32')

        # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although
        # scaling is done later, its result is necessary for the psychoacoustic model and calculation of
        # sound pressure levels.
        for ch in range(wav_file.get_num_of_ch()):
            # scfindices[ch, :] = get_scalefactors(subband_samples[ch, :, :], params.table.scalefactor) TODO
            # subband_bit_allocation[ch, :] = psycho.model1(wav_file.audio[ch].ordered(), params, scfindices) TODO
            pass

        subband_samples_quantized = np.zeros(subband_samples.shape, dtype='uint32')
        for ch in range(wav_file.get_num_of_ch()):
            for sb in range(tables.N_SUBBANDS):
                QCa = wav_file.get_table().qca[subband_bit_allocation[ch, sb] - 2]
                QCb = wav_file.get_table().qcb[subband_bit_allocation[ch, sb] - 2]
                scf = wav_file.get_table().scalefactor[scfindices[ch, sb]]
                ba = subband_bit_allocation[ch, sb]
                for ind in range(tables.FRAMES_PER_BLOCK):
                    # subband_samples_quantized[ch, sb, ind] = quantization.quantization(subband_samples[ch, sb, ind], scf, ba, QCa, QCb) TODO
                    pass
        # Forming output bitsream and appending it to the output file.
        # bitstream_formatting(outmp3file, params, subband_bit_allocation, scfindices, subband_samples_quantized) TODO
