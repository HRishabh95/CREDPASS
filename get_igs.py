import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from scipy import spatial

model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb', device='cuda')

fd=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/mayoclinic_sentences.csv',sep='\t',names=['disease_name','text'],header=None)
#fd=fd.iloc[1:,:]
ts_clef=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_BM25_top_sentences.csv',sep='\t')

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
        qsimil.append([vec[0], 1 - spatial.distance.cosine(query_vec, vec[1])])
    d = sorted(qsimil,
               key=lambda x: -x[1])
    top_2_topics = [d[0][0], d[1][0]]
    return top_2_topics


def get_top_portal_sentences(doc_vec,por_vec):
    finsimil=[]
    for jj,dvec in enumerate(doc_vec):
        porsimil=[]
        for kk,pvec in enumerate(por_vec):
            porsimil.append([dvec[0],pvec[0],1-spatial.distance.cosine(dvec[1],pvec[1])])
        d = sorted(porsimil,
                   key=lambda x: -x[2])
        d = d[:10]
        simi=0
        for dd in d:
            simi+=float(dd[-1])
        fsimi=simi/len(d)
        finsimil.append([d[0][0],fsimi])
    simi=0
    for finsimi in finsimil:
        simi+=float(finsimi[-1])
    fsimi=simi/len(finsimil)

    return finsimil

dis_vect=[]
for ii,fd_rows in fd.iterrows():
    dis_vect.append([ii,get_vectors(fd_rows['disease_name'])])

final_simi=[]
for ii,rows in ts_clef.iterrows():
    print(ii)
    if ii==10:
        break
    tmp_list=[]
    query=get_vectors(rows['query'])
    top_topics=get_most_similar_disease(dis_vect,query)

    doc_vec = []
    for text in rows['top_sentences'].split("."):
        if len(text)>1:
            doc_vec.append([text,get_vectors(text)])

    por_vec = []
    for top_topic in top_topics[:-1]:
        for text in fd.iloc[top_topic]['text'].split("."):
            if len(text)>1:
                por_vec.append([text,get_vectors(text)])

    final_simi.append(get_top_portal_sentences(doc_vec,por_vec))