import os
import pandas as pd
import re
import string
PUNCTUATIONS = string.punctuation.replace(".",'')
import random
import numpy as np
import json
random.seed(47)

def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def trec_generate(f):
  df = pd.DataFrame(f, columns=['docno', 'text'])
  return df


#Bm25 retrived
docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_biobert_nltk_correct.csv',sep='\t')

qrels=pd.read_csv('/home/ubuntu/rupadhyay/2020-derived-qrels//misinfo-qrels.2aspects.useful-credible',sep=' ',
                  names=['qid','Q0','docno','t','c'])
qrels['label']=qrels['t']+qrels['c']-1

docs_merged=pd.merge(qrels,docs_100,on=['docno','qid'])
docs_merged.dropna(inplace=True)
docs_merged=docs_merged[['qid','query','docno','text','label','score','top_sentences']]


test=0.3
json_required=True
qids = docs_merged['qid'].unique()
test_set=int(len(qids) * test)
train_set=len(qids)-test_set
train_qid=random.sample(list(qids),k=train_set)
test_qid=[i for i in qids if i not in train_qid]
final_dataset_train = []
final_dataset_test = []

for qid in train_qid:
    final_dataset_train.append(docs_merged[docs_merged['qid'] == qid])
for qid in test_qid:
    final_dataset_test.append(docs_merged[docs_merged['qid'] == qid])


pd.concat(final_dataset_train).to_csv('./train_qid_clef_bm25.csv', sep=';')
pd.concat(final_dataset_test).to_csv('./test_qid_clef_bm25.csv', sep=';')

if json_required:
    final_dataset_train = json.loads(pd.concat(final_dataset_train).to_json(orient='records'))
    final_dataset_test = json.loads(pd.concat(final_dataset_test).to_json(orient='records'))

    with open("train_qid_trec_bm25_pass.jsonl", "w") as f:
        for item in final_dataset_train:
            f.write(json.dumps(item) + "\n")

    with open("test_qid_trec_bm25_pass.jsonl", "w") as f:
        for item in final_dataset_test:
            f.write(json.dumps(item) + "\n")


