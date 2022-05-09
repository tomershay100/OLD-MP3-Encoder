import os
import sys

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
    # params = wav_file.get_params()
    # main(inwavfile, outmp3file, bitrate)
