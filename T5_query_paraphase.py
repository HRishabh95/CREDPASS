from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_sentence_paraphraser")
t5_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_sentence_paraphraser")

paraphase=[]
for ii,qrel in qrels.iterrows():
    inputs = t5_tokenizer.encode("summarize: "+qrel['query'] , return_tensors="pt", max_length=512, padding='max_length', truncation=True)

    summary_ids = t5_model.generate(inputs,num_beams=int(2),no_repeat_ngram_size=4,length_penalty=2.0,
                                        min_length=4,
                                        max_length=20,
                                        early_stopping=True)

    output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    paraphase.append([qrel['query'],output])


from transformers import pipeline

summarization=pipeline('summarization',model='GanjinZero/biobart-base')
print(summarization("Can vitamin C cure Covid-19?",max_length=10))