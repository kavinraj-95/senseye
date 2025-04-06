import speech_recognition as sr
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import soundfile as sf

scaler = StandardScaler()
model = SVC()

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    mfccs_processed = np.mean(mfccs.T, axis=0)
    chroma_processed = np.mean(chroma.T, axis=0)
    mel_processed = np.mean(mel.T, axis=0)

    features = np.hstack([mfccs_processed, chroma_processed, mel_processed])
    return features

def analyze_voice_emotion():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak something...")
        audio = recognizer.listen(source)

        # Save the audio temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        # Extract features and analyze emotions
        features = extract_features("temp_audio.wav")
        features = scaler.transform([features])
        emotion_prediction = model.predict(features)
        
        # Print the predicted emotion (this is a placeholder)
        print("Detected emotion:", emotion_prediction[0])

if __name__ == "__main__":
    analyze_voice_emotion()