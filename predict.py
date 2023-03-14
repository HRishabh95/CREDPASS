import os.path
import datasets
import pandas as pd
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import mkdir_p

cred_score=False

data='trec'

if data=='trec':
    #TREC
    docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_biobert_nltk_correct_sent.csv',sep='\t')
    dataset=docs_100[['qid','query','docno','top_sentences','score']]
    dataset.columns=['qid','query','docno','text','score']
    dataset['text']=dataset['text'].str.replace("\t"," ")
    dataset.dropna(subset=['text'], inplace=True)


#CLEF

if data=='clef':
    docs_100=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/clef2020_BM25_biobert_nltk_correct_sent_5.csv',sep='\t')
    dataset=docs_100[['qid','query','docno','top_sentences','score']]
    dataset.columns=['qid','query','docno','text','score']
    dataset['text']=dataset['text'].str.replace("\t"," ")
    dataset.dropna(subset=['text'],inplace=True)
#8 , 24


for batch_number, batch in enumerate([2,3,4,5,6,8,10,16,20]):
    for epoch_number, epoch in enumerate([1,2,3,4,5,6,7,8,9]):
        print(batch,epoch)
        model_path = f'''./cross_encoder_MRR_biobert-v1.1_clef_20/cross_encoder_{epoch}_{batch}_passage/'''
        file_path = f'''./cross_encoder_MRR_biobert-v1.1_clef_20/results_evaluation/cross_encoder_biobert_{epoch}_{batch}_passage.csv'''
        mkdir_p("/".join(file_path.split("/")[:-1]))
        if not os.path.isfile(file_path):
            if os.path.isfile(model_path+"/config.json"):
                model = CrossEncoder(model_path, num_labels=1, max_length=510)

                result_df=[]
                for ii,row in tqdm(dataset.iterrows()):
                    score=model.predict([row['query'],row['text']])
                    result_df.append([row['qid'],0,row['docno'],score])

                result_df=pd.DataFrame(result_df,columns=['qid','Q0','docno','score'])
                qids = np.unique(result_df.qid.values)
                sorted_dfs=[]
                for qid in qids:
                    if qid!=28 and qid!=33 and qid!=35:
                        qid_df=result_df.loc[result_df['qid']==qid]
                        sorted_qid_df=qid_df.sort_values('score',ascending=False).reset_index()
                        sorted_qid_df['n_rank']=1
                        for i in sorted_qid_df.index:
                            sorted_qid_df.at[i,'n_rank']=i+1
                        sorted_dfs.append(sorted_qid_df)

                sorted_qid_df_concat=pd.concat(sorted_dfs)
                result_df=sorted_qid_df_concat[['qid','Q0','docno','n_rank','score']]
                result_df.columns=['qid','Q0','docno','rank','score']
                if cred_score:
                    result_df['experiment']=f'''cross_encoder_bert_{epoch}_{batch}_bm25'''
                else:
                    result_df['experiment'] = f'''cross_encoder_biobert_{epoch}_{batch}_passage_10'''

                result_df.to_csv(file_path,sep=' ',index=None,header=None)
