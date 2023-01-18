import torch
import subprocess
medical_words=open('medical_term.txt',encoding='utf-8').readlines()
medical_words=[i.lower().replace("\n","") for i in medical_words]

## Hard cutoff
##TODO add different type of method for masking
def medical_term(term):
    if term.lower() in medical_words:
        return 1
    else:
        return 0

def get_attention_mask(tokenized_text,indexed_tokens):
    attention_mask = []
    for tup in zip(tokenized_text, indexed_tokens):
        attention_mask.append(medical_term(tup[0]))
    return attention_mask


def show_gpu(msg):
    """
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')

def get_vector(model,tokens_tensor,hidden_layer=True,hidden_layer_number=2,add=True):
    outputs = model(tokens_tensor)
    last_hidden_state = outputs[0]
    word_embed = last_hidden_state
    if hidden_layer:
        hidden_states = outputs[2]
        if hidden_layer_number!=0 and add:
            word_embed = torch.stack(hidden_states[hidden_layer_number:]).sum(0)
        elif hidden_layer_number!=0:
            word_embed= torch.cat([hidden_states[i] for i in range(1,hidden_layer_number+1)], dim=-1)
        elif add:
            word_embed = torch.stack(hidden_states).sum(0)
        else:
            word_embed = torch.cat([hidden_states[i] for i in range(1, len(hidden_states)+1)], dim=-1)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return word_embed[0]

def get_s_vector(vec,attention_mask):
    attention_mask = torch.transpose(torch.tensor(attention_mask), 0, -1)
    multi_vec=torch.mul(torch.transpose(vec, 0, 1), attention_mask)
    return multi_vec.sum(1)


def get_sentence_vector(marked_text,model,tokenizer,dynamic_attention=True,hidden_layer=False,hidden_layer_number=2,add=True):
    #marked_text="[CLS] " + text + " [SEP]"
    #model_name="dmis-lab/biobert-v1.1"
    tokenized_text = tokenizer.tokenize(marked_text)
    tokenized_text=tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    if dynamic_attention:
        attention_mask = get_attention_mask(tokenized_text, indexed_tokens)
    else:
        attention_mask = [1] * len(indexed_tokens)
        
    tokens_tensor = torch.tensor([indexed_tokens])
    vec = get_vector(model, tokens_tensor,hidden_layer=hidden_layer,hidden_layer_number=hidden_layer_number,add=add)

    sen_vec = get_s_vector(vec, attention_mask)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return sen_vec


# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#model = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True)
# marked_text = "[CLS] " + "cancer kills" + " [SEP]"
#
# #Mean of Last 2 hidden layer with dynamic attention
# sen_vec_1_type_1=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=True,hidden_layer=True,
#                                      hidden_layer_number=2,add=True)
#
# #Mean of last 2 hidden layer with static attention
# sen_vec_1_type_2=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=True,
#                                      hidden_layer_number=2,add=True)
#
# #Last layer with dynamic attention
# sen_vec_1_type_3=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=True,hidden_layer=False,
#                                      hidden_layer_number=2,add=True)
#
#
# #Last layer with static attention
# sen_vec_1_type_4=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=False,
#                                      hidden_layer_number=2,add=True)
#
#
# #Last 4 layer with static attention
# sen_vec_1_type_5=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=True,
#                                      hidden_layer_number=4,add=True)
#
# marked_text = "[CLS] " + "cancer safe" + " [SEP]"
#
# #Mean of Last 2 hidden layer with dynamic attention
# sen_vec_2_type_1=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=True,hidden_layer=True,
#                                      hidden_layer_number=2,add=True)
#
# #Mean of last 2 hidden layer with static attention
# sen_vec_2_type_2=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=True,
#                                      hidden_layer_number=2,add=True)
#
# #Last layer with dynamic attention
# sen_vec_2_type_3=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=True,hidden_layer=False,
#                                      hidden_layer_number=2,add=True)
#
#
# #Last layer with static attention
# sen_vec_2_type_4=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=False,
#                                      hidden_layer_number=2,add=True)
#
#
# #Last 4 layer with static attention
# sen_vec_2_type_5=get_sentence_vector(marked_text,tokenizer,model,dynamic_attention=False,hidden_layer=True,
#                                      hidden_layer_number=4,add=True)
#
#
# cos=torch.nn.CosineSimilarity(dim=0)
# print(cos(sen_vec_1_type_1,sen_vec_2_type_1))
# print(cos(sen_vec_1_type_2,sen_vec_2_type_2))
# print(cos(sen_vec_1_type_3,sen_vec_2_type_3))
# print(cos(sen_vec_1_type_4,sen_vec_2_type_4))
# print(cos(sen_vec_1_type_5,sen_vec_2_type_5))
#
