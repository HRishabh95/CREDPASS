import os.path
from os import listdir,cpu_count
from os.path import join

import pandas as pd
import warc
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import random
from multiprocessing import Pool
random.seed(49)

trec_data_path='/home/ubuntu/shared/TREC2020/01/'
list_gzs=listdir(trec_data_path)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

list_gzs=random.sample(list_gzs, 20)

def get_html(warc_file):
    file_path='/home/ubuntu/shared/TREC2020/01_extracted/'
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


file_content=[]
for list_gz in list_gzs:
    warc_files=warc.open(join(trec_data_path,list_gz))
    warc_files=[i for i in warc_files]
    # processing (multiprocessing)
    #result = p.map(get_html, warc_files[:10])
    for warc_file in warc_files:
        result = get_html(warc_file)
    # for warc_file in warc_files:
    #     if warc_file.type=='response':
    #         try:
    #             r=requests.get(warc_file.url,headers=headers)
    #             soup = BeautifulSoup(r.content).get_text().replace('\n', ' ')
    #             soup = ' '.join(soup.split())
    #             lang=detect(soup)
    #             if lang=='en':
    #                 file_content.append([warc_file.header.record_id.split(":")[-1].split(">")[0], soup])
    #         except:
    #             print('URL not reachable')

