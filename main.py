import os
import re
import numpy as np
import tqdm
from multiprocessing import Pool
import librosa
import librosa.display

from feature_mfcc import get_features

AUDIO_PATH = "speech-accent-archive/recordings/recordings/"

r = re.compile("english*|spanish*|arabic*|mandarin*|french*|korean*|portuguese*|russian*|dutch*")
r1 = re.compile("english*|spanish*|arabic*")
if __name__ == "__main__":
    filelist = os.listdir("speech-accent-archive/recordings/recordings")
    filelist = list(filter(r1.match,filelist))
    pool = Pool(processes=7)
    features = pool.map(get_features, filelist)
    pool.close()
    pool.join()

    all_features = np.vstack([f[0] for f in features])
    all_accents = [f[1] for f in features]
    # Normalize the data
    all_features = (all_features - np.mean(all_features, axis=0))/(np.std(all_features, axis=0))
    unique_accents, mapped_values = np.unique(all_accents, return_inverse=True)
    mapped_values = np.expand_dims(mapped_values, axis=1)
    final_dataset = np.hstack((all_features, mapped_values))
<<<<<<< HEAD
    # np.save('final_dataset_top9_delta.npy', final_dataset)
    # np.save('unique_accents_top9_delta.npy', unique_accents)
    ## For flattened data
    np.save('mfcc_delta_data.npy', final_dataset)
    np.save('unique_accents_mfcc_delta.npy', unique_accents)
    np.save('filelist_order.npy', filelist)
=======
    np.save('final_dataset_delta_std_3.npy', final_dataset)
    np.save('unique_accents_delta_std_3.npy', unique_accents)
>>>>>>> 15f1be4a5e7961da80267752152231dde6455260
