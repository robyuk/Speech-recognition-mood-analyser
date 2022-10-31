from speech_recognition import Recognizer, AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

file=input("\nEnter filename of audio file: ")

analyse = SentimentIntensityAnalyzer()

recognise=Recognizer()

with AudioFile(file) as audiofile:
  audio = recognise.record(audiofile)

text=recognise.recognize_google(audio)

if text == "":
  print(f'No speech in the audiofile {file}')
  print(text)
  print(analyse.polarity_scores(text))