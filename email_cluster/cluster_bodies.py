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

X = df_vector_bodies[df_vector_bodies.columns].values
print(X.shape)

labels = HDBSCAN(min_samples=10).fit_predict(X)
print(labels.shape)

#labels = db.labels_
np.save("subjectHDBSCAN.npy", labels)
gc.collect()

labels = np.load("subjectHDBSCAN.npy")
print(labels.shape)
gc.collect()

#from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.metrics import silhouette_score
#score = cython_silhouette_score(X, labels, chunksize= 20000) 
score = silhouette_score(X=X, labels=labels, metric= "euclidean", sampling= 0.1, n_jobs= 6)
print(score)
gc.collect()

