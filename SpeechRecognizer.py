import speech_recognition as sr #importing speech recognisation<br>

Audio_file=("sample_viah.wav")# Use audio file(viah) as source<br>

r = sr.Recognizer() # intialise speech regoniser<br>
with sr.AudioFile(Audio_file) as source:
    audio=r.record(source) # reads audio file
    
try:
    print("Speech Recognistion Sucessfull. The file Contained" + r.recognize_google(audio)) #converts spoken word to text<br>
except sr.UnknownValueError: #If unable to understand audio<br>
    print("Google Speech Recognition Could Not Understand Audio ")
except sr.RequestError:#If unable to fetch results from server og google speech API<br>
    print("Unable to Fetch Results")
