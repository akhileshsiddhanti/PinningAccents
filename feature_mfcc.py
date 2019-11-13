import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import describe
import re

AUDIO_PATH = "speech-accent-archive/recordings/recordings/"

def get_features(path):
    accent = re.split('\d',path)[0]
    print(accent)
    waveform, sampling_rate = librosa.load(AUDIO_PATH + path)
    stft = np.abs(librosa.stft(waveform))
    mfcc = np.mean(librosa.feature.mfcc(waveform,sampling_rate,n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(waveform, sr=sampling_rate), axis=1)
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sampling_rate), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(waveform), sr=sampling_rate), axis=1)

    features = np.hstack([mfcc, chroma, mel, spec_bw, contrast, tonnetz])
    return features, accent


if __name__ == '__main__':

    path = "speech-accent-archive/recordings/recordings/english1.mp3"
    others = get_features(path)

