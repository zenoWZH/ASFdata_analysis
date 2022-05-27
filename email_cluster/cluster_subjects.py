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

df_vector_subjects = pd.DataFrame(np.load("vector30_subjects.npy"))

from cuml.cluster import DBSCAN

X = df_vector_subjects[df_vector_subjects.columns]

db = DBSCAN(eps=5, min_samples=20).fit(X)

labels = db.labels_
#df_all_commiters["subject_labels"] = labels

from sklearn import metrics 

score = metrics.silhouette_score(X, labels) 
print(score)

np.save("subjectlabels.npy", labels)
gc.collect()