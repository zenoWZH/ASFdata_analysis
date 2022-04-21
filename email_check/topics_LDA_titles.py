import os
import numpy as np
import pandas as pd 
#import modin.pandas as pd
from tqdm import tqdm
import re
import datetime
from p_tqdm import p_map
from functools import partial
import gc

from gensim import corpora

from gensim.models.ldamulticore import LdaMulticore

time_resolution = '1'
lk_path = '/mnt/data0/lkyin/'
c_path = '../network_data'+str(time_resolution)+'/commits/'
e_path = '../network_data'+str(time_resolution)+'/emails/'
mix_path = '../network_data'+str(time_resolution)+'/mix/'

df = pd.read_csv('./emails_preprocessed.csv')

all_graduated = np.load('../all_graduated.npy').tolist()
#all_graduated = [x.lower() for x in all_graduated]
all_retired = np.load('../all_retired.npy').tolist()
#all_retired = [x.lower() for x in all_retired]

print('grouping by project...')
dfgroup = df.groupby(['project_name'])

titles_graduated = list()
bodys_graduated = list()
#proj_graduated = list()
for proj in all_graduated:
    try:
        this_proj = dfgroup.get_group(proj.lower())
    except BaseException as err:
        print(err)
        continue
    #for x in range(len(this_proj)):
    #    proj_graduated.append(proj.lower())
    titles_graduated.extend(this_proj['subject'].values)
    bodys_graduated.extend(this_proj['body'].values)


titles_retired = list()
bodys_retired = list()

for proj in all_retired:
    try:
        this_proj = dfgroup.get_group(proj.lower())
    except BaseException as err:
        print(err)
        continue

    titles_retired.extend(this_proj['subject'].values)
    bodys_retired.extend(this_proj['body'].values)


# Train LDA model with titles of graduated emails
texts = [str(x).split() for x in titles_graduated]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


ldamodel = LdaMulticore(corpus, num_topics=20, id2word = dictionary, passes=10, random_state = 1, workers=6) 
ldamodel.save("./title_graduated_model.lda")
print(ldamodel.print_topics(num_topics=20, num_words=3))

# Train LDA model with titles of retired emails
texts = [str(x).split() for x in titles_retired]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


ldamodel = LdaMulticore(corpus, num_topics=20, id2word = dictionary, passes=10, random_state = 1, workers=6)
ldamodel.save("./title_retired_model.lda")
print(ldamodel.print_topics(num_topics=20, num_words=3))