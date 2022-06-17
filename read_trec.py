import os.path
from os import listdir,cpu_count
from os.path import join
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import random
from multiprocessing import Pool
random.seed(49)

trec_data_path='/home/ubuntu/shared/TREC2020/01/'
list_gzs=listdir(trec_data_path)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

list_gzs=random.sample(list_gzs,5)
def get_html(warc_file):
    file_path='/home/ubuntu/shared/TREC2020/01/'
    file_id = warc_file.header.record_id.split(":")[-1].split(">")[0]
    file_path=f'''{file_path}{file_id}.csv'''
    if not os.path.isfile(file_path):
        if warc_file.type == 'response':
            try:
                r=requests.get(warc_file.url,headers=headers)
                soup = BeautifulSoup(r.content).get_text().replace('\n', ' ').replace("\t",' ')
                soup = ' '.join(soup.split())
                lang=detect(soup)
                if lang=='en':
                    print(file_path)
                    pd.DataFrame([file_id, soup]).to_csv(file_path,sep='\t')
                    return None
            except:
               return None


from warcio.archiveiterator import ArchiveIterator
file_content=[]
for ii,list_gz in enumerate(list_gzs):
    print(ii,len(list_gzs))
    with open(join(trec_data_path,list_gz), 'rb') as stream:
        for record in ArchiveIterator(stream, arc2warc=True):
            if record.rec_type == 'response':
                file_id=record.rec_headers.get_header('WARC-Record-ID').split(":")[-1].split(">")[0]
                content=record.content_stream().read()
                try:
                    if len(content)>0:
                        soup = BeautifulSoup(content).get_text().replace('\n', ' ').replace("\t", ' ')
                        soup = ' '.join(soup.split())
                        lang = detect(soup)
                        if lang == 'en':
                            file_content.append([file_id, soup])
                except:
                    print('Error')


trec_file=pd.DataFrame(file_content)
trec_file.columns=['docno','text']
for col in trec_file.columns:
    if trec_file[col].dtype==object:
        trec_file[col]=trec_file[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))

trec_file.to_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/TREC_HM_1M.csv''',sep='\t',index=False)

import pandas as pd
first_df=pd.read_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/TREC_HM_1M.csv''',sep='\t')
second_df=pd.read_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/TREC_HM_2M.csv''',sep='\t')
third_df=pd.read_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/TREC_HM_3M.csv''',sep='\t')
fourth_df=pd.read_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/TREC_HM_4M.csv''',sep='\t')

trec_df=pd.concat([first_df,second_df,third_df,fourth_df])

trec_df.to_csv("/home/ubuntu/rupadhyay/CREDPASS/TREC_1M.csv",sep='\t')

import pandas as pd
trec_df=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/TREC_1M.csv",sep='\t',index_col=0)
labeled_file=pd.read_csv('/tmp/pycharm_project_889/labeled_trec20201.csv',sep='\t',header=None,index_col=0)
labeled_file.columns=['docno','text']
final_file=pd.concat([trec_df,labeled_file]).drop_duplicates().reset_index(drop=True)
final_file.to_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_1M_labeled.csv',sep='\t')