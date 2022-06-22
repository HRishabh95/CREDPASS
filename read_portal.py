import pandas as pd
from allennlp.predictors.predictor import Predictor
import re,string
fd=pd.read_csv('/tmp/pycharm_project_889/mayo_clinic/disease_data.csv',sep='\t',index_col=None)
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")


PUNCTUATIONS = string.punctuation
PUNCTUATIONS = PUNCTUATIONS.replace(".",'')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?%\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()


def process_p(name):
    if type(rows[name]) is str:
        texts=clean_en_text(rows[name])
        if texts:
            if len(texts)>3:
                return [predictor.coref_resolved(texts)]
            else:
                return []
        else:
            return []
    else:
        return []

def process_symp_il(rows):
    if type(rows['symp_il']) is str:
        symp_il=[]
        sympsi=rows['symp_il'].split("..")
        for symps in sympsi:
            for symp in symps.split("."):
                if len(symp.split())<8 and len(symp)!=0:
                    symp_il.append(f'''{symp} are symptoms of {rows['disease_name']}''')
        #symp_il=".".join(symp_il)
        return symp_il
    else:
        return []


def process_causes_il(rows):
    if type(rows['causes_il']) is str:
        symp_il=[]
        sympsi=rows['causes_il'].split("..")
        for symps in sympsi:
            for symp in symps.split("."):
                if len(symp.split())<8 and len(symp)!=0:
                    symp_il.append(f'''{rows['disease_name']} can be caused by {symp}''')
        #symp_il=".".join(symp_il)
        return symp_il
    else:
        return []


def process_treat_il(rows):
    if type(rows['treat_il']) is str:
        symp_il = []
        sympsi = rows['treat_il'].split("..")
        for symps in sympsi:
            for symp in symps.split("."):
                if len(symp.split())<8 and len(symp)!=0:
                    symp_il.append(f'''{rows['disease_name']} can be treated by {symp}''')
        #symp_il = ".".join(symp_il)
        return symp_il
    else:
        return []

def process_prev_il(rows):
    if type(rows['pre_il']) is str:
        symp_il = []
        sympsi = rows['pre_il'].split("..")
        for symps in sympsi:
            for symp in symps.split("."):
                if len(symp.split())<8 and len(symp)!=0:
                    symp_il.append(f'''{rows['disease_name']} can be prevented by {symp}''')
        #symp_il = ".".join(symp_il)
        return symp_il
    else:
        return []

def process_diag_il(rows):
    if type(rows['diag_il']) is str:
        symp_il = []
        sympsi = rows['diag_il'].split("..")
        for symps in sympsi:
            for symp in symps.split("."):
                if len(symp.split())<8 and len(symp)!=0:
                    symp_il.append(f'''{rows['disease_name']} can be diagnosed by {symp}''')
        return symp_il
    else:
        return []


all_data=[]
for ii,rows in fd.iterrows():
    print(ii)
    intro=process_p('intro')
    symp_p=process_p('symp_p')
    causes_p=process_p('causes_p')
    diag_p=process_p('diag_p')
    prev_p=process_p('prev_p')
    treat_p=process_p('treat_p')
    symp_il=process_symp_il(rows)
    causes_il=process_causes_il(rows)
    treat_il=process_treat_il(rows)
    prev_il=process_treat_il(rows)
    diag_il=process_diag_il(rows)
    data=intro+symp_p+causes_p+diag_p+treat_p+prev_p+symp_il+causes_il+treat_il+prev_il+diag_il
    datas='.'.join(data)
    datas=datas+'.'
    datas=datas.replace("..",'.')
    all_data.append([rows['disease_name'],datas])

pd.DataFrame(all_data).to_csv('/home/ubuntu/rupadhyay/CREDPASS/mayoclinic_sentences.csv',sep='\t',index=False,header=False)