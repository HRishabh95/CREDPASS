import pandas as pd
import re
import string

import torch
from allennlp.predictors.predictor import Predictor

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
  text = re.sub(r"[^A-Za-z0-9(),.!?%\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

import torch
first_stage_rank=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_clean.csv',sep='\t')
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

#first_stage_rank['text']=first_stage_rank['text'].apply(clean_en_text)
#first_stage_rank.to_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_clean.csv',sep='\t',index=None)
corref=[]
file_path='/tmp/corref/'
for ii,row in first_stage_rank.iterrows():
  print(ii)
  f=open(f'''{file_path}/{row['docno']}_{ii}.txt''','w')
  f.write(row['text'])
  f.close()
  corref.append(predictor.coref_resolved(row['text']))

first_stage_rank['text_coref']=corref
#first_stage_rank['text']=first_stage_rank['text'].apply(predictor.coref_resolved)
first_stage_rank.to_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_coref.csv',sep='\t',index=None)


