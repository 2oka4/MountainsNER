# -*- coding: utf-8 -*-


from transformers import BertForTokenClassification, pipeline, BertTokenizerFast
import pandas as pd

"""To test the model we use another Hugging Face class: pipeline. It makes the process of getting the predicted result easier as we do not need to do data preprocessing steps manually."""

label2id = {'O': 0, 'B-mount': 1, 'I-mount': 2} #labels are necessary for the model
id2label = {0: 'O', 1: 'B-mount', 2: 'I-mount'}

dir = "./model_save" #directory with presaved model and tokenizer settings

"""First of all, we need to load the pretrained parameters of model and tokenizer"""

tokenizer = BertTokenizerFast.from_pretrained(dir)  #we use BertTokenizerFast instead of BertTokenizer because the BertTokenizer cannot handle aggregation_strategy="first" in a pipeline
model = BertForTokenClassification.from_pretrained(dir,
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)

pipe_ner = pipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first")
#aggregation strategy is defined because we did worpiece tokenization. So the output of the model originally is also pieces of the words and their labels. aggregation_strategy="simple" changes the output from worpieces to words
pipe_ner("Ben Nevis is a popular hiking destination, with 150,000 people a year visiting the peak.")

"""We need to write custom function to extract mountain(s) name(s) from model prediction"""

def get_mountains_from_predictions(predictions):
    # Combine the words to reconstruct the sentence
    words = [pred['word'] for pred in predictions]
    sentence = " ".join(words)

    # Remove extra spaces (e.g., around punctuation)
    sentence = sentence.replace(" ##", "")  # Handle subword tokens if they appear
    sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" !", "!")
    return sentence

get_mountains_from_predictions(pipe_ner("Ben Nevis is a popular hiking destination, with 150,000 people a year visiting the peak."))

"""Now we will try to use our model on the new dataset"""

df = pd.read_csv("mountain_sentences.csv")

df

df["Mountain"] = 0

for i in range(len(df)):
  df.loc[i, "Mountain"] = get_mountains_from_predictions(pipe_ner(df.iloc[i,0]))

df.head(15)

df.tail(15)

"""As we can see, the model has no problems finding names of mountains in these sentences. I will save this dataset."""

df.to_csv("predictions.csv")
