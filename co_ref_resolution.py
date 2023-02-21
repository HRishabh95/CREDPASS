import os.path

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

docs=first_stage_rank.drop_duplicates(subset=['docno']).reset_index(drop=True)
doc=docs[['docno','text']]

#first_stage_rank['text']=first_stage_rank['text'].apply(clean_en_text)
#first_stage_rank.to_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_clean.csv',sep='\t',index=None)
corref=[]
file_path='/tmp/corref/'
for ii,row in doc.iterrows():
  print(ii)
  try:
    corref_file_path=f'''{file_path}/{row['docno']}.txt'''
    #if not os.path.isfile(corref_file_path):
    f=open(corref_file_path,'w')
    coref=predictor.coref_resolved(row['text'])
    f.write(coref)
    f.close()
    corref.append([row['docno'],coref])
    # else:
    #   f = open(f'''{file_path}/{row['docno']}.txt''', 'r').read()
    #   corref.append([row['docno'],predictor.coref_resolved(row['text'])])
  except:
    print("Eorrir")

doc_corref=pd.DataFrame(corref,columns=['docno','text_coref'])
docs_coref=pd.merge(doc_corref,doc,on='docno')
#doc['text_coref']=corref
coref_df=pd.merge(first_stage_rank,docs_coref,how='left',on='docno')
#first_stage_rank['text']=first_stage_rank['text'].apply(predictor.coref_resolved)
coref_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_coref.csv',sep='\t',index=None)


