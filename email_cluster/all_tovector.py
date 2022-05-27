import os
import numpy as np
import pandas as pd 
#import modin.pandas as pd
from tqdm import tqdm
import re
import datetime
from functools import partial
import gc

df_all_commiters = pd.read_csv("./commiters_emails.csv")


from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(['First do it', 'then do it right', 'then do it better']))

### Here max_lenth=100, subject using 25
body_subjects = bc.encode(df_all_commiters["body"].apply(lambda x: str(x)).values.tolist())

np.save("vector200_bodies.npy", body_subjects)

