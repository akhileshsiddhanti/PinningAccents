import librosa


def getMFCC(path):
    waveform, sampling_rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(waveform,sampling_rate,n_mfcc=20)

    return mfcc


if __name__ == '__main__':

    path = "speech-accent-archive/recordings/recordings/indonesian1.mp3"
    mfcc = getMFCC(path)
    
    print (mfcc)
    print (len(mfcc))