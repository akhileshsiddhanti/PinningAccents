import feature_mfcc


if __name__ == "__main__":
    
    path = "speech-accent-archive/recordings/recordings/indonesian1.mp3"
    mfcc = feature_mfcc.getMFCC(path)
    print (mfcc)