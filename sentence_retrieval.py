import pandas as pd
import numpy as np
#from sentence_transformers import SentenceTransformer
import torch
import math
from scipy import spatial
from AS_BERT import get_sentence_vector,show_gpu
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
#model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb', device='cuda')

#model_name = "bert-base-uncased"
#model_name = "dmis-lab/biobert-v1.1"
#model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
model_name = 'deepset/covid_bert_base'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)


#model_name = 'allenai/biomed_roberta_base'
#tokenizer = RobertaTokenizer.from_pretrained(model_name)
#model=RobertaModel.from_pretrained(model_name, output_hidden_states=True)
#model_name = '/home/ubuntu/rupadhyay/CREDPASS/TREC-150k-biobert-10epochs/'
#model_name = "dmis-lab/biobert-v1.1"

model.to('cuda')

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

top_10_sents=[]
for ii,rows in first_stage_rank.iterrows():
    print(ii)
    tmp_list=[]
    vect=[]
    try:
        #query_vec=get_vectors(rows['query'])
        query_vec = get_sentence_vector(rows['query'],model,tokenizer,dynamic_attention=False,hidden_layer=False)
        for kk,text in enumerate(rows['text'].split(".")):
            if 4<len(text.split())<60:
                #vect.append([text,get_vectors(text)])
                vect.append([text,get_sentence_vector(text,model,tokenizer,dynamic_attention=False,hidden_layer=False)])
        simil=[]
        for jj,vec in enumerate(vect):
            #simil.append([vec[0],1 - spatial.distance.cosine(query_vec, vec[1])])
            simil.append([vec[0],1 - spatial.distance.cosine(query_vec.cpu().detach().numpy(), vec[1].cpu().detach().numpy())])
        d=sorted(simil,
               key=lambda x: -x[1])
        top_10_d=d[:10]
        flat_list_text=".".join([sublist[0] for sublist in top_10_d if sublist[1] > 0.1])
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
top_10_sents_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_top_sentences.csv',sep='\t',index=False)

#clef2020_BM25_top_sentences_att_SciBERT --- Fine tunned model bert-uncased
#clef2020_BM25_top_sentences --- Biobert


# esimil=[]
# for ii,vec in enumerate(vect):
#     esimil.append([vec[0],1 - spatial.distance.euclidean(query_vec, vec[1])])
# e=sorted(esimil,
#        key=lambda x: -x[1])
#
#
# edsimil=[]
# for ii,vec in enumerate(vect):
#     edsimil.append([vec[0],mapfunct(np.linalg.norm(query_vec-vec[1]))])
# e=sorted(edsimil,
#        key=lambda x: -x[1])
