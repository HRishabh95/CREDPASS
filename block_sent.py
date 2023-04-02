import re
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.append("/tmp/pycharm_project_635/CogLTX")
from CogLTX.buffer import Buffer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import language_tool_python
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

cnt=0
vect=[]
d='Potential harms of chloroquine, hydroxychloroquine and azithromycin for treating COVID-19. Potential harms of chloroquine, hydroxychloroquine and azithromycin for treating COVID-19. Chloroquine, hydroxychloroquine and azithromycin are being used to treat and prevent COVID-19 despite weak evidence for effectiveness, and physicians and patients should be aware of the drugs’ potentially serious adverse events, states a review in CMAJ (Canadian Medical Association Journal). Physicians and patients should be aware of several rare but potentially life-threatening adverse effects of chloroquine and hydroxychloroquine, says Dr. David Juurlink, Division of Clinical Pharmacology and Toxicology, Sunnybrook Health Sciences Centre, and a senior scientist at ICES. The review provides an overview of potential harms associated with these drugs as well as their management based on the best available evidence. Potential adverse effects include: Cardiac arrhythmias Hypoglycemia Neuropsychiatric effects, such as agitation, confusion, hallucinations and paranoia Interactions with other drugs Metabolic variability (some people metabolize chloroquine and hydroxychloroquine poorly and a small percentage metabolize them rapidly, which affects the response to treatment) Overdose (chloroquine and hydroxychloroquine are highly toxic in overdose and can cause seizures, coma and cardiac arrest) Drug shortages (patients with autoimmune disorders such as rheumatoid arthritis, lupus and other chronic diseases, who take hydroxychloroquine to treat these conditions could have problems accessing this drug) The review summarizes the poor quality of evidence suggesting that these treatments might be beneficial in patients with COVID-19 and cautions that it is possible that these treatments could worsen the disease. Despite optimism (in some, even enthusiasm) for the potential of chloroquine or hydroxychloroquine in the treatment of COVID-19, little consideration has been given to the possibility that the drugs might negatively influence the course of disease, says Dr. Juurlink. This is why we need a better evidence base before routinely using these drugs to treat patients with COVID-19. Provided by Canadian Medical Association Journal.'
dbuf, cnt = Buffer.split_document_into_blocks(tokenizer_b.tokenize(d),tokenizer_b, cnt, hard=False)

for kk, text in enumerate(dbuf):
    text = text.__str__()
    if 4 < len(text.split()) < 80:
        vect.append([text])
