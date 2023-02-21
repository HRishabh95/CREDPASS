import os
import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init()
import sys
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
  text = re.sub(r"[^A-Za-z0-9(),.!?%\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def make_docs_cerche(dfs):
  docs=[]
  for ii,df in dfs.iterrows():
    if df['text']:
      docs.append({'id':df['docno'],'title':'','article':clean_en_text(df['text'])})
  return docs

config={'TREC':{'file_path':'/home/ubuntu/rupadhyay/CREDPASS/TREC2020_1M_labeled_clean.csv',
                'index_path':'/home/ubuntu/rupadhyay/CREDPASS/trec2020_bm25',
                'topics':'/home/ubuntu/rupadhyay/dataset/TREC/topics.csv'},
        'CLEF':{'file_path':'/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled_clean.csv',
                'index_path':'/home/ubuntu/rupadhyay/CREDPASS/clef2020_bm25',
                'topics':'/home/ubuntu/rupadhyay/CREDPASS/clef_topics.csv'}}

if len(sys.argv)>3:
    data_set=sys.argv[1]
    indexing=sys.argv[2]
else:
    data_set = 'TREC'
    indexing = False

if indexing:
    index_doc=pd.read_csv(config[data_set]['file_path'],sep='\t')
    index_doc=index_doc.dropna(subset=['text'])
    index_doc.drop_duplicates(subset=['text'],inplace=True)

    index_path=config[data_set]['index_path']
    index_doc=index_doc[['docno','text']]
    if not os.path.exists(f'''{index_path}/data.properties'''):
        indexer = pt.DFIndexer(index_path, overwrite=True, verbose=True, Threads=8)
        indexer.setProperty("termpipelines", "PorterStemmer") # Removes the default PorterStemmer (English)
        indexref3 = indexer.index(index_doc["text"], index_doc)
    else:
        indexref3 = pt.IndexRef.of(f'''{index_path}/data.properties''')

indexref3 = pt.IndexRef.of(f'''{config[data_set]['index_path']}/data.properties''')
BM251 = pt.BatchRetrieve(indexref3, num_results=500, controls = {"wmodel": "BM25"})


topics=pt.io.read_topics(config[data_set]['topics'],format='singleline')
results=BM251.transform(topics)


results = results[~results["qid"].isin(['28'])]
results['Q0']=0
result=results[['qid','Q0','docno','rank','score']]
results.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25.csv',sep=' ', index=None, header=None)

qrels_path='/home/ubuntu/rupadhyay/CREDPASS/trec_qrels_top.csv'
qrels = pt.io.read_qrels(qrels_path)
eval = pt.Utils.evaluate(results,qrels,metrics=["ndcg"], perquery=True)
# eval.to_csv('/home/ubuntu/rupadhyay/CREDPASS/clef_map.csv',sep='\t')

docs=pd.read_csv(config[data_set]['file_path'],sep='\t')
docs=docs[['docno','text']]
merged_results=pd.merge(results,docs,on=['docno'])
#merged_results.to_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_BM25.csv',sep='\t',index=None)
#merged_results['text']=merged_results['text'].apply(clean_en_text)
merged_results.to_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv',sep='\t',index=None)

