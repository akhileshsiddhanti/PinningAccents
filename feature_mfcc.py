import librosa
import librosa.display
import matplotlib.pyplot as plt


def getMFCC(path):
    waveform, sampling_rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(waveform,sampling_rate,n_mfcc=20)
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(mfcc, x_axis='time')
    plt.figure()
    librosa.display.waveplot(waveform, sr=sampling_rate)
    plt.title('Waveform')
    plt.show()
    return mfcc


if __name__ == '__main__':

    path = "speech-accent-archive/recordings/recordings/indonesian1.mp3"
    mfcc = getMFCC(path)

    print (mfcc)
    print (len(mfcc))
