# Transform multiple YouTube video URLs to text scripts with language detection.
# Author: Javed Ali (www.javedali.net)

# import required modules
import os
import whisper
from langdetect import detect
from pytube import YouTube

# Function to open a file
def startfile(fn):
    os.system('open %s' % fn)

# Function to create and open a txt file
def create_and_open_txt(text, filename):
    # Create and write the text to a txt file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    startfile(filename)

f = open(r"C:\Users\Dell\OneDrive\Desktop\audio-to-text-transcription-main (1)\Course3.txt", "r")
video_urls = f.readlines()

# Output directory for audio & text files
download_directory = r"C:\Users\Dell\OneDrive\Desktop\audio-to-text-transcription-main (1)\audio-to-text-transcription-main\YoutubeAudios\Course3_dnld"
output_directory = r"C:\Users\Dell\OneDrive\Desktop\audio-to-text-transcription-main (1)\audio-to-text-transcription-main\YoutubeAudios\Transcribed_text3"

# Loop through each video URL
for url in video_urls:
    # Create a YouTube object from the URL
    yt = YouTube(url)

    # Get the audio stream
    audio_stream = yt.streams.filter(only_audio=True).first()

    # Download the audio stream
    filename = f"audio_{yt.video_id}.mp3"
    audio_stream.download(output_path=download_directory, filename=filename)

    #print(f"Audio downloaded to {output_directory}/{filename}")

    # Load the base model and transcribe the audio
    model = whisper.load_model("base")
    result = model.transcribe(os.path.join(download_directory, filename))
    transcribed_text = result["text"]

    # Detect the language
    #language = detect(transcribed_text)
    #print(f"Detected language for {url}: {language}")

    # Create and open a txt file with the text
    output_filename = f"output_{yt.video_id}.txt"
    create_and_open_txt(transcribed_text, os.path.join(output_directory, output_filename))
