from transformers import BertTokenizer, BertModel
import torch

medical_words=open('medical_term.txt').readlines()
medical_words=[i.lower().replace("\n","") for i in medical_words]

## Hard cutoff
##TODO add different type of method for masking
def medical_term(term):
    if term.lower() in medical_words:
        return 1
    else:
        return 0.5

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_attention_mask(tokenized_text,indexed_tokens):
    attention_mask = []
    attention_mask.append(0)
    for tup in zip(tokenized_text[1:-1], indexed_tokens[1:-1]):
        attention_mask.append(medical_term(tup[0]))
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    attention_mask.append(0)
    return attention_mask

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
    return word_embed[0]

def get_sentence_vector(vec,attention_mask):
    attention_mask = torch.transpose(attention_mask, 0, -1)
    multi_vec=torch.mul(torch.transpose(vec[0], 0, 1), attention_mask)
    return multi_vec.sum(1)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True)



marked_text = "[CLS] " + "cancer is deadly" + " [SEP]"
# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)


indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.

attention_mask=get_attention_mask(tokenized_text,indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])
model.eval()
vec=get_vector(model,tokens_tensor)

sen_vec=get_sentence_vector(vec,attention_mask)
