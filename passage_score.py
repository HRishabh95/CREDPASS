import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy import spatial
from AS_BERT import get_sentence_vector,show_gpu
from transformers import BertTokenizer, BertModel
import torch

# model_name = "dmis-lab/biobert-v1.1"
model_name = "pritamdeka/S-BioBert-snli-multinli-stsb"
model = SentenceTransformer(model_name, device='cuda')
#model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# tokenizer = BertTokenizer.from_pretrained(model_name)
#model_name = '/home/ubuntu/rupadhyay/CREDPASS/TREC-150k-biobert-10epochs/'
# model = BertModel.from_pretrained(model_name, output_hidden_states=True)
model.to('cuda')


fd=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/mayoclinic_sentences.csv',sep='\t',names=['disease_name','text'],header=None)
ts_clef=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_biobert_nltk_correct.csv',sep='\t')
data='TREC'


def get_vectors(texts):
    chuck_vecs=[]
    for i in range(0, len(texts.split()), 512):
        texts_512 = " ".join(texts.split()[i:i + 512])
        sen_embeddings = model.encode(texts_512)
        chuck_vecs.append(sen_embeddings)
    return np.mean(chuck_vecs, axis=0)

def get_most_similar_disease(dis_vec,query_vec):
    qsimil = []
    for jj, vec in enumerate(dis_vec):
        qsimil.append([vec[0], 1 - spatial.distance.cosine(query_vec.cpu().detach().numpy(), vec[1].cpu().detach().numpy())])
        #qsimil.append([vec[0], 1 - spatial.distance.cosine(query_vec, vec[1])])
    d = sorted(qsimil,
               key=lambda x: -x[1])
    top_2_topics = [d[0][0], d[1][0]]
    return top_2_topics


def get_top_portal_sentences(doc_vec,por_vec):
    finsimil=[]
    for jj,dvec in enumerate(doc_vec):
        porsimil=[]
        for kk,pvec in enumerate(por_vec):
            # porsimil.append([dvec[0],pvec[0],1-spatial.distance.cosine(dvec[1].cpu().detach().numpy(),pvec[1].cpu().detach().numpy())])
            porsimil.append([dvec[0], pvec[0], 1 - spatial.distance.cosine(dvec[1][0],
                                                                           pvec[1][0])])
            #porsimil.append([dvec[0], pvec[0], 1 - spatial.distance.cosine(dvec[1],pvec[1])])
        d = sorted(porsimil,
                   key=lambda x: -x[2])
        d = d[:10]
        simi=0
        for dd in d:
            simi+=float(dd[-1])
        fsimi=simi/len(d)
        finsimil.append([d[0][0],fsimi])
    doc_simi=0
    weights_10 = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025]
    #weights_10 = [0.25, 0.15, 0.125, 0.125, 0.1, 0.05, 0.05, 0.05,0.05,0.05]
    for ii,finsimi in enumerate(finsimil):
        doc_simi+=float(finsimi[-1])*weights_10[ii]
    fsimi=doc_simi

    return len(d),fsimi

dis_vect=[]
for ii,fd_rows in fd.iterrows():
    #dis_vect.append([ii,get_vectors(fd_rows['disease_name'])])
    #dis_vect.append([ii,get_sentence_vector(fd_rows['disease_name'],model,tokenizer,dynamic_attention=False,hidden_layer=False)])
    dis_vect.append([ii, model.encode([fd_rows['disease_name']])])

final_simi=[]
for ii,rows in ts_clef.iterrows():
    print(ii)
    try:
        tmp_list=[]
        #query=get_vectors(rows['query'])
        # query=get_sentence_vector(rows['query'], model, tokenizer, dynamic_attention=False,hidden_layer=False)
        query=model.encode([rows['query']])
        if data=='TREC':
            top_topics=[418]
        else:
            top_topics=get_most_similar_disease(dis_vect,query)
        doc_vec = []
        if type( rows['top_sentences']) is str:
            for text in rows['top_sentences'].split("\t"):
                if 4 < len(text.split()) <= 60:
                    #doc_vec.append([text,get_vectors(text)])
                    # doc_vec.append([text,get_sentence_vector(text,model,tokenizer,dynamic_attention=False,hidden_layer=False)])
                    doc_vec.append([text,model.encode([text])])
            por_vec = []
            for top_topic in top_topics:
                for text in fd.iloc[top_topic]['text'].split("."):
                    if 4 < len(text.split()) <= 60:
                        #por_vec.append([text, get_vectors(text)])
                        #por_vec.append([text,get_sentence_vector(text,model,tokenizer,dynamic_attention=False,hidden_layer=False)])
                        por_vec.append([text,model.encode([text])])
            number_of_retrieval,igs_score=get_top_portal_sentences(doc_vec,por_vec)
            final_simi.append([
                rows['qid'],
                rows['docid'],
                rows['docno'],
                rows['rank'],
                rows['score'],
                rows['query'],
                rows['text'],
                number_of_retrieval,
                igs_score])

    except:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        show_gpu(f'GPU memory usage after loading training objects:')

top_10_sents_df=pd.DataFrame(final_simi)
top_10_sents_df.columns=['qid','docid','docno','rank','score','query','text','noret','igs_score']
top_10_sents_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_SRet_10_PRet_10.csv',sep='\t',index=False)

#clef2020_BM25_SRet_10_PRet_10_first_weight_fine_tunned.csv
#clef2020_BM25_SRet_10_PRet_10_SentTrans_Biobert_3top.csv === trec
