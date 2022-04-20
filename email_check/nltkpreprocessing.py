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

from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('omw-1.4')
lemma = WordNetLemmatizer()

mystopwords = ['re', 'cc', 'fwd', 'fyi']
mystopwords = stopwords.words('english').append(mystopwords)

def stopandlemma(txt):
    txt = str(txt).lower()
    tokens = word_tokenize(txt)
    tokens = [lemma.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if(x not in stopwords.words('english'))]
    # very slow
    #gc.collect()
    return ' '.join(tokens)

def ff(num):
    return num*num

from multiprocessing import Pool

if __name__ == '__main__' :

    df = pd.read_csv('./emails_nomarks.csv')
    print("Start Preprocessing")
    #df['subject'] = df['subject'].map(lambda s: stopandlemma(str(s).lower()))
    #df.to_csv('./emails_preprocessed_2.csv', index_label= None)
    #print("Titles Done!!!")
    # Too Slow
    #df['body'] = df['body'].map(lambda s: stopandlemma(str(s).lower()))

    bodies = df['body'].values
    # Single process
    #for txt in tqdm(bodies):
    #    txt = stopandlemma(str(txt).lower())

    # multi process
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        result = list(pool.imap(ff, range(100), chunksize=4))
        print(result)
    pool.close()
    pool.join()

    with Pool(6) as p:
        result = list(tqdm(p.imap(stopandlemma, bodies, chunksize=6), total=len(bodies), desc="Multiprocess preprocessing"))
    p.close()
    p.join()

    df['body'] = pd.Series(bodies)
    df.to_csv('./emails_preprocessed_3.csv', index_label= None)
    print("Bodies Done!!")