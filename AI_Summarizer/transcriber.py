import whisper 
from pytube import YouTube
from pytube.exceptions import AgeRestrictedError
from pydub import AudioSegment
import json
import io
import tempfile
import progressbar

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

f = open(f'/Users/uladzimircharniauski/Documents/GetBetterHead/video_ids.json')
ids = json.load(f)

model = whisper.load_model("base")

def get_audio_stream(video_id):

    try:

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()

        return audio_stream
    except AgeRestrictedError:
        return None

# Function to convert audio stream to WAV format
def stream_to_wav(audio_stream):
    try:
        buffer = io.BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        audio_segment = AudioSegment.from_file(buffer, format="mp4")
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer
    except audio_stream == None:
        return None

def transcribe_videos():

    transcript_array = []

    bar = progressbar.ProgressBar(maxval=120)
    i=0
    bar.start()

    for item in ids:

        bar.update(i + 1) 
        
        i+=1

        audio = get_audio_stream(item['videoId'])

        wav_buffer = stream_to_wav(audio)

        try:

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                tmp_file.write(wav_buffer.read())
                tmp_file.seek(0)
                transcript_text = model.transcribe(tmp_file.name, fp16=False)

            transcript_array.append(transcript_text['text'])

        except wav_buffer == None:
            pass

    bar.finish()

    return transcript_array

#Loading it in a JSON file:

def save_texts_to_json(transcripts, filename='youtube_texts.json'):
    with open(filename, 'w') as file:
        json.dump(transcripts, file, indent=4)

if __name__ == "__main__":

    print('Transcription process has started. Please be patient, as it may take a lot of time to transcribe all of your videos.')

    transcripts = transcribe_videos()

    save_texts_to_json(transcripts, filename='youtube_texts.json')

    print('Generating JSON file with transcripts is successful!')

