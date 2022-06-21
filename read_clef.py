from bs4 import BeautifulSoup
import pandas as pd
dataset_path='/home/ubuntu/rupadhyay/dataset/CLEF2020/'
from os import listdir
from os.path import isfile, join
from langdetect import detect
import numpy as np

onlyfiles=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/Clef_files.csv")
file_content=[]
for ii,file in onlyfiles.iterrows():
    if ii%10000==0:
        print(ii)
    soup=BeautifulSoup(open(file['filename']).read()).get_text().replace('\n',' ')
    soup=' '.join(soup.split())
    try:
        if len(soup.split())>20:
            lang = detect(soup)
            if lang == 'en':
                file_content.append([file['filename'].split("/")[-1],soup])
    except:
        print("Text not good")

clef_file=pd.DataFrame(file_content)
clef_file.columns=['docno','text']
for col in clef_file.columns:
    if clef_file[col].dtype==object:
        clef_file[col]=clef_file[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))

clef_file.to_csv(f'''/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M.csv''',sep='\t',index=False)

labeled_file=pd.read_csv('/tmp/pycharm_project_889/labeled_clef2020.csv',sep='\t')

final_file=pd.concat([clef_file,labeled_file]).drop_duplicates().reset_index(drop=True)
final_file.to_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled.csv',sep='\t')