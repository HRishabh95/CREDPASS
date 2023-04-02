import re
import numpy as np
from scipy import spatial
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import torch
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from AS_BERT import get_sentence_vector,show_gpu
import sys
sys.path.append("/tmp/pycharm_project_635/CogLTX")
from CogLTX.buffer import Buffer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import language_tool_python
from tqdm import tqdm
import numpy as np

tool = language_tool_python.LanguageTool('en-US')

def cal_grammar_score(sentence):

    scores_word_based = []
    matches = tool.check(sentence)
    count_errors = len(matches)
    # only check if the sentence is correct or not
    scores_sentence_based_sentence=np.min([count_errors, 1])
    scores_word_based_sentence=count_errors
    word_count = len(sentence.split())
    sum_count_errors_word_based = np.sum(scores_word_based_sentence)
    score_word_based = 1 - (sum_count_errors_word_based / word_count)
    scores_word_based=score_word_based
    sum_count_errors_sentence_based = np.sum(scores_sentence_based_sentence)
    scores_sentence_based = 1 - np.sum(sum_count_errors_sentence_based / 1)

    return np.mean(scores_sentence_based)


from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

args = TTSettings(num_beams=5, min_length=1)

def clean(data):
    tmp_doc = []
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            words = words.replace('</s>','')
            c = re.sub(r'[<>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc

DEFAULT_MODEL_NAME = 'roberta-base'
tokenizer_b = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)


#model_name = "bert-base-uncased"
model_name = "emilyalsentzer/Bio_ClinicalBERT"
#model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#model_name = 'deepset/covid_bert_base'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name, output_hidden_states=True)
model = SentenceTransformer(model_name)

#model.to('cuda')

def get_vectors(texts):
    chuck_vecs=[]
    for i in range(0, len(texts.split()), 512):
        texts_512 = " ".join(texts.split()[i:i + 512])
        sen_embeddings = model.encode(texts_512)
        chuck_vecs.append(sen_embeddings)
    return np.mean(chuck_vecs, axis=0)


def mapfunct(x, type='exp', n=0.2):
    """
    Map 0-inf to 1-0 with some function
    Type:
    """
    if type == 'inverse':
        return 1 / (1 + n * x)
    if type == 'arctan':
        return 1 - 2 / math.pi * math.atan(n * x)
    if type == 'exp':
        return (1 / (1 + n)) ** x
    else:
        raise (NotImplementedError("Function not implemented"))

def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)

#first_stage_rank=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_BM25_clean_100.csv',sep='\t')
first_stage_rank=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_clean_100.csv',sep='\t')

cnt= 0
top_10_sents=[]
for ii,rows in first_stage_rank.iterrows():
    print(ii)
    tmp_list=[]
    vect=[]
    try:
        #query_vec=get_vectors(rows['query'])
        #query_vec = get_sentence_vector(rows['query'],model,tokenizer,dynamic_attention=False,hidden_layer=False)
        query_vec=model.encode([rows['query']])[0]
        for i in range(0,len(rows['text'].split()),350):
            dbuf, cnt = Buffer.split_document_into_blocks(tokenizer_b.tokenize(" ".join(rows['text'].split()[i:i+400])), tokenizer_b, cnt, hard=False)
            for kk,text in enumerate(dbuf):
                text = text.__str__()
                text=happy_tt.generate_text(text).text
                text = clean(text)
                #if cal_grammar_score(text):
                if 4<len(text.split())<80:
                    #vect.append([text,get_vectors(text)])
                    #vect.append([text,get_sentence_vector(text,model,tokenizer,dynamic_attention=False,hidden_layer=False)])
                    vect.append([text, model.encode([text])])
            simil=[]
            for jj,vec in enumerate(vect):
                #simil.append([vec[0],1 - spatial.distance.cosine(query_vec, vec[1])])
                # simil.append([vec[0],1 - spatial.distance.cosine(query_vec.cpu().detach().numpy(), vec[1].cpu().detach().numpy())])
                simil.append([vec[0],1 - spatial.distance.cosine(query_vec, vec[1][0])])

            d=sorted(simil,
                   key=lambda x: -x[1])
            top_10_d=d[:10]
            flat_list_text="\t".join([sublist[0] for sublist in top_10_d if sublist[1] > 0.1])
            flat_list_score=",".join([str(sublist[1]) for sublist in top_10_d if sublist[1] > 0.1])

            top_10_sents.append([
                rows['qid'],
                rows['docid'],
                rows['docno'],
                rows['rank'],
                rows['score'],
                rows['query'],
                rows['text'],
                flat_list_text,
                flat_list_score])
    except:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        show_gpu(f'GPU memory usage after loading training objects:')

top_10_sents_df=pd.DataFrame(top_10_sents)
top_10_sents_df.columns=['qid','docid','docno','rank','score','query','text','top_sentences','top_scores']
top_10_sents_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_top_blocks_biobert_1.2.csv',sep='\t',index=False)
