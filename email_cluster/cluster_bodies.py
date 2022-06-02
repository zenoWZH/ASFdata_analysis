import os
import numpy as np
import pandas as pd 
#import modin.pandas as pd
from tqdm import tqdm
import re
import datetime
from functools import partial
import gc

#df_all_commiters = pd.read_csv("./commiters_emails.csv")
#vector_subjects = np.load("vector30_subjects.npy")

df_vector_bodies = pd.DataFrame(np.load("vector200_bodies.npy"))

from cuml.cluster import HDBSCAN
##################################################################
X = df_vector_bodies[df_vector_bodies.columns].values
print(X.shape)

for i in range(1,5):
    print("min samples= ", i)
    labels = HDBSCAN(min_samples= i).fit_predict(X)
    print(labels.shape)

    np.save("subjectHDBSCAN_s"+str(i)+".npy", labels)
    gc.collect()
#################################################################
X = np.load("vector200_bodies.npy")
labels = np.load("subjectHDBSCAN_s5.npy")
print(labels.shape)
gc.collect()

#from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.metrics import silhouette_score
#score = cython_silhouette_score(X, labels, chunksize= 20000) 
score = silhouette_score(X=X, labels=labels, metric= "euclidean", sample_size= 10000, n_jobs= 6)
print(score)
gc.collect()
