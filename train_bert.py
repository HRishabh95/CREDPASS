from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import pandas as pd
from langdetect import detect
import re
import string


tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = BertForNextSentencePrediction.from_pretrained('dmis-lab/biobert-v1.1')


## TREC index_col=0
final_file=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled_clean.csv',sep='\t')
texts=final_file.sample(n=150000,random_state=49)['text'].values
clean=False

text=[]
for txts in texts:
    try:
        if detect(txts)=='en':
            text.append(txts)
    except:
        print("Error in Text")


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
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

if clean:
    textt=[clean_en_text(i) for i in text]
else:
    textt=text

bag = [item for sentence in textt for item in sentence.split('.') if item != '']
bag_size = len(bag)
print(bag_size)


import random

sentence_a = []
sentence_b = []
label = []

for paragraph in textt:
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, bag_size-1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)


inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['labels'] = torch.LongTensor([label]).T




########## Move to Dataset

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


dataset = MeditationsDataset(inputs)


#### Dataloader

loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

model.zero_grad()

##### Activation


from transformers import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=5e-6)


#### Training

from tqdm import tqdm  # for our progress bar

epochs = 10

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


model.save_pretrained('/home/ubuntu/rupadhyay/CREDPASS/CLEF-150k-biobert-10epochs/')
tokenizer.save_pretrained('/home/ubuntu/rupadhyay/CREDPASS/CLEF-150k-biobert-10epochs/')