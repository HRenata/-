# Convert all *.wav files in this folder to .wav encoding for transcription

find . -iname "*.wav" | wc

for wavfile in `find . -iname "*.wav"`
do
    ffmpeg -y -f wav -i $wavfile -ab 64k -ac 1 -ar 16000 -f wav "${wavfile}"
done