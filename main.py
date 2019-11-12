import feature_mfcc
import os
import re
import numpy as np
import tqdm

AUDIO_PATH = "speech-accent-archive/recordings/recordings/"

r = re.compile("english*|spanish*|arabic*|mandarin*|french*|korean*|portuguese*|russian*|dutch*")
if __name__ == "__main__":
    filelist = os.listdir("speech-accent-archive/recordings/recordings")
    filelist = list(filter(r.match,filelist))
    mfcc_array = []
    accents = []
    for file in tqdm.tqdm(filelist):
    	accent = re.split('\d',file)[0]
    	path = AUDIO_PATH + file
    	mfcc = feature_mfcc.getMFCC(path)
    	if len(mfcc_array) == 0:
    		mfcc_array = mfcc
    		accents = accent
    	else:
    		mfcc_array = np.vstack((mfcc_array, mfcc))
    		accents = np.vstack((accents, accent))

    # Normalize the data
    mfcc_array = (mfcc_array - np.mean(mfcc_array,axis=0))/(np.std(mfcc_array, axis=0))
    # final_dataset = np.hstack((mfcc_array, accents))
    unique_accents, mapped_values = np.unique(accents, return_inverse=True)
    mapped_values = np.expand_dims(mapped_values, axis=1)
    final_dataset = np.hstack((mfcc_array, mapped_values))
    print(final_dataset.shape)
    np.save('mfcc_dataset.npy', final_dataset)
    np.save('accent_map.npy', unique_accents)
# np.save('outfilemfcctop9', mfcc_array)
