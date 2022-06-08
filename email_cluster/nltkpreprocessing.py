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

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.text import Text
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('omw-1.4')
class EmailPreprocessor:

    def __init__(self):
        self.lemma = WordNetLemmatizer()
        self.mystopwords = stopwords.words('english')
        self.mystopwords.extend(['re', 'cc', 'fwd', 'fyi'])
        self.processcounter = 0
        print("Processor Initialized")
        print("Stop Words:", self.mystopwords)
        # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
        self.pat_letter = re.compile(r'[^a-zA-Z \']+')
        # 还原常见缩写单词
        self.pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
        self.pat_s = re.compile("(?<=[a-zA-Z])\'s") # 找出字母后面的字母
        self.pat_s2 = re.compile("(?<=s)\'s?")
        self.pat_not = re.compile("(?<=[a-zA-Z])n\'t") # not的缩写
        self.pat_would = re.compile("(?<=[a-zA-Z])\'d") # would的缩写
        self.pat_will = re.compile("(?<=[a-zA-Z])\'ll") # will的缩写
        self.pat_am = re.compile("(?<=[I|i])\'m") # am的缩写
        self.pat_are = re.compile("(?<=[a-zA-Z])\'re") # are的缩写
        self.pat_ve = re.compile("(?<=[a-zA-Z])\'ve") # have的缩写
        self.citereply = re.compile("> wrote:\n")

    def replace_abbreviations(self, text):
        new_text = text
        new_text = self.pat_letter.sub(' ', text).strip().lower()
        new_text = self.pat_is.sub(r"\1 is", new_text)
        new_text = self.pat_s.sub("", new_text)
        new_text = self.pat_s2.sub("", new_text)
        new_text = self.pat_not.sub(" not", new_text)
        new_text = self.pat_would.sub(" would", new_text)
        new_text = self.pat_will.sub(" will", new_text)
        new_text = self.pat_am.sub(" am", new_text)
        new_text = self.pat_are.sub(" are", new_text)
        new_text = self.pat_ve.sub(" have", new_text)
        new_text = new_text.replace('\'', ' ')
        return new_text
    
    def get_wordnet_pos(self, treebank_tag):

        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return ''

    def onlyreply(self, txt):
        txt = str(txt)
        posi = txt.find("> wrote:\n")
        if posi == -1:
            return txt
        else:
            while txt[posi]!="\n" and posi>0:
                posi-= 1
            if posi>0:
                return txt[:posi]
            else:
                return txt  


    def stopandlemma(self, txt):
        self.processcounter+=1
        txt = self.onlyreply(txt)
        #print(txt)
        if txt == None :
            return ''
        newtxt = []
        txt = self.replace_abbreviations(str(txt).lower())
        sentences = sent_tokenize(txt)
        for sent in sentences:
            words = []
            tokens = word_tokenize(sent)
            tokensandtags = nltk.pos_tag(tokens)
            for wordwithtag in tokensandtags:
                word = wordwithtag[0]
                pos = self.get_wordnet_pos(wordwithtag[1])
                if pos and len(word)<30 and word not in self.mystopwords:
                    word = self.lemma.lemmatize(word, pos)
                    words.append(word)
            newtxt.extend(words)

        if self.processcounter >10000 :
            gc.collect()
            self.processcounter = 0

        return ' '.join(newtxt)

#def ff(num):
#    return num*num

from multiprocessing import Pool

if __name__ == '__main__' :
    print("Reading Data")
    df = pd.read_csv('./edges_emails.csv')
    print("Start Preprocessing Email Titles")

    epre = EmailPreprocessor()
    
    bodies = df['subject'].values
    with Pool(6) as p:
        result = list(tqdm(p.imap(epre.stopandlemma, bodies, chunksize=6), total=len(bodies), desc="Multiprocess preprocessing Titles"))
    p.close()
    p.join()
    df['subject'] = pd.Series(result)

    print("Start Preprocessing Email Bodies")

    bodies = df['body'].values
    with Pool(6) as p:
        result = list(tqdm(p.imap(epre.stopandlemma, bodies, chunksize=6), total=len(bodies), desc="Multiprocess preprocessing Bodies"))
    p.close()
    p.join()

    df['body'] = pd.Series(result)
    print("Bodies Done!!")
    df.to_csv('./edges_emails_preprocessed.csv', index= None)
    print("Saved!")
