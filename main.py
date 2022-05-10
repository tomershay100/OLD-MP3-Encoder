import os
import sys

from MP3_Encoder import MP3Encoder
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
    encoder = MP3Encoder(wav_file)
    encoder.encode()
