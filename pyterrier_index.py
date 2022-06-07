import pandas as pd
index_doc=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled.csv',sep='\t',index_col=0)

import pyterrier as pt
import os
import sys
import collections
import pandas as pd
import numpy as np
import pickle
if not pt.started():
  pt.init()

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"

import re
import string
PUNCTUATIONS = string.punctuation
PUNCTUATIONS = PUNCTUATIONS.replace(".",'')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()


def trec_generate(f):
  df = pd.DataFrame(f, columns=['docno', 'text'])
  return df


import os

wic_data = '/home/ricky/Documents/PhDproject/dataset/trec/trec_20_wic_top10_en_nd.csv'
index_path='/home/ricky/Documents/PhDproject/indexs/BM25_baseline_nds'
f = pickle.load(open(wic_data, 'rb'))
df_docs = trec_generate(f)
