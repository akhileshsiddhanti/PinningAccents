import feature_mfcc
import os
import numpy as np

if __name__ == "__main__":
    filelist = os.listdir("speech-accent-archive/recordings/recordings")
    path_1 = "speech-accent-archive/recordings/recordings/"+filelist[0]
    mfcc_array = feature_mfcc.getMFCC(path_1)
    filelist.pop(0)
    for i in filelist:
        path = "speech-accent-archive/recordings/recordings/"+i
        mfcc = feature_mfcc.getMFCC(path)
        np.append(mfcc_array,mfcc)
np.save('outfilemfcc', mfcc_array)
