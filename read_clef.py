from bs4 import BeautifulSoup
import pandas as pd
dataset_path='/home/ubuntu/rupadhyay/dataset/CLEF2020/'
from os import listdir
from os.path import isfile, join
from langdetect import detect

# onlyfiles = [f'''{dataset_path}{f}/{j}''' for f in listdir(dataset_path) for j in listdir(join(dataset_path,f))]
# df_files=pd.DataFrame(onlyfiles,columns=['filename'])
# onlyfiles=df_files['filename'].sample(n=1200000,random_state=49)
# onlyfiles.to_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef_files.csv',index=False)

onlyfiles=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/Clef_files.csv")
file_content=[]
for ii,file in onlyfiles.iterrows():
    if ii%100==0:
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


file_con=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M.csv',sep='\t')

labeled_file=pd.read_csv('/tmp/pycharm_project_889/labeled_clef2020.csv',sep='\t')

final_file=pd.concat([file_con,labeled_file]).drop_duplicates().reset_index(drop=True)
final_file.to_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled.csv',sep='\t')