import feature_mfcc
import os
import re
import numpy as np

r = re.compile("english*|spanish*|arabic*|mandarin*|french*|korean*|portuguese*|russian*|dutch*")
if __name__ == "__main__":
    filelist = os.listdir("speech-accent-archive/recordings/recordings")
    filelist1 = list(filter(r.match,filelist))
    path_1 = "speech-accent-archive/recordings/recordings/"+filelist1[0]
    mfcc_array = feature_mfcc.getMFCC(path_1)
    filelist1.pop(0)
    for i in filelist1:
        path = "speech-accent-archive/recordings/recordings/"+i
        mfcc = feature_mfcc.getMFCC(path)
        np.append(mfcc_array,mfcc)
np.save('outfilemfcctop9', mfcc_array)
