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
	bw_array = []
	for file in tqdm.tqdm(filelist):
		accent = re.split('\d',file)[0]
		path = AUDIO_PATH + file
		mfcc = feature_mfcc.getMFCC(path)
		bw = feature_mfcc.spectral_bw(path)
		if len(mfcc_array) == 0:
			mfcc_array = mfcc
			accents = accent
			bw_array = bw
		else:
			mfcc_array = np.vstack((mfcc_array, mfcc))
			accents = np.vstack((accents, accent))
			bw_array = np.vstack((bw_array,bw))

	# Normalize the data
	mfcc_array = (mfcc_array - np.mean(mfcc_array,axis=0))/(np.std(mfcc_array, axis=0))
	bw_array = (bw_array - np.mean(bw_array,axis=0))/(np.std(bw_array, axis=0))
	mfcc_array = np.hstack((mfcc_array,bw_array))
	# final_dataset = np.hstack((mfcc_array, accents))
	unique_accents, mapped_values = np.unique(accents, return_inverse=True)
	mapped_values = np.expand_dims(mapped_values, axis=1)
	final_dataset = np.hstack((mfcc_array, mapped_values))
	print(final_dataset.shape)
	np.save('mfcc_dataset.npy', final_dataset)
	np.save('accent_map.npy', unique_accents)
