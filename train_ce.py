import math
import os.path
from utils import mkdir_p
import datasets
import numpy as np
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers.evaluation import InformationRetrievalEvaluator

cred_score=False

data='clef'
if data=='trec':
    dataset = datasets.load_dataset("json", data_files={"train": ["train_qid_trec_bm25_pass_5.jsonl"]})
else:
    dataset = datasets.load_dataset("json", data_files={"train": ["train_qid_clef_bm25_pass_20.jsonl"]})


train_samples = []
for row in tqdm(dataset['train']):
    if cred_score:
        c_score = f'''{row['c_score']:.4f} [SEP]''' # Acc:0.7, F1- 0.69
    else:
        c_score = ''
    train_samples.append(InputExample(
        texts=[row['query'], c_score+row['top_sentences'].replace("\t"," ")], label=float(row['label'])
    ))

if data=='trec':
    dataset = datasets.load_dataset("json", data_files={"test": ["test_qid_trec_bm25_pass_5.jsonl"]})
else:
    dataset = datasets.load_dataset("json", data_files={"test": ["test_qid_clef_bm25_pass_20.jsonl"]})

dataset_pos = dataset.filter(
    lambda x: True if x['label'] == 0 else False
)

dataset_neg = dataset.filter(
    lambda x: True if x['label'] == 1 else False
)

dev_sample={}
for i in dataset_pos['test']:
    if i['qid'] not in dev_sample:
        dev_sample[i['qid']]={
            'query': i['query'],
            'positive': set(),
            'negative':set()
        }
    if i['qid'] in dev_sample:
        if cred_score:
            c_score = f'''{i['c_score']:.4f} [SEP]'''  # Acc:0.7, F1- 0.69
        else:
            c_score = ''
        dev_sample[i['qid']]['positive'].add(c_score+i['top_sentences'].replace("\t"," "))
    for j in dataset_neg['test']:
        if j['qid']==i['qid']:
            if cred_score:
                c_score = f'''{j['score']:.4f} [SEP]'''  # Acc:0.7, F1- 0.69
            else:
                c_score = ''
            dev_sample[i['qid']]['negative'].add(c_score+j['top_sentences'])



torch.manual_seed(47)
model_name = '/tmp/pycharm_project_447/biobert-v1.1'
# model_name = '/tmp/pycharm_project_447/bert-base-uncased_v2'

best_score=0
model_path = f'''./cross_encoder_MRR_{model_name.split("/")[-1]}'''
result_folder = f'''{model_path}/result'''
mkdir_p(result_folder)
if cred_score:
    result_file = f'''{model_path}/result/c_score.csv'''
else:
    result_file = f'''{model_path}/result/no_c_score.csv'''

for batch_number, batch in enumerate([2,3,4,6,8,10,12]):
    for epoch_number, epoch in enumerate([1,2,3,4,5,6,7,8,9]):
        print(batch, epoch)
        train_batch_size = batch

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        evaluator = CERerankingEvaluator(dev_sample, name='train-eval')
        if cred_score:
            model_save_path = f'''./cross_encoder_MRR_{model_name.split("/")[-1]}_{data}_10/cross_encoder_{epoch}_{train_batch_size}_''' + 'c_score_only'
            mkdir_p(model_save_path)
        else:
            model_save_path = f'''./cross_encoder_MRR_{model_name.split("/")[-1]}_{data}_20/cross_encoder_{epoch}_{train_batch_size}_passage'''
            mkdir_p(model_save_path)

        if not os.path.isfile(model_save_path + "/pytorch_model.bin"):
            print("Training")
            warmup_steps = math.ceil(len(train_dataloader) * epoch * 0.1)  # 10% of train data for warm-up

            model = CrossEncoder(model_name, num_labels=1, max_length=512,
                                 automodel_args={'ignore_mismatched_sizes': True})

            # Train the model
            model.fit(train_dataloader=train_dataloader,
                      evaluator=evaluator,
                      epochs=epoch,
                      evaluation_steps=2000,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      use_amp=True,
                      )
