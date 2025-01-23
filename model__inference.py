# -*- coding: utf-8 -*-


from transformers import BertTokenizer, BertForTokenClassification, pipeline

"""To test the model we use another Hugging Face class: pipeline. It makes the process of getting the predicted result easier as we do not need to do data preprocessing steps manually."""

label2id = {'O': 0, 'B-mount': 1, 'I-mount': 2} #labels are necessary for model
id2label = {0: 'O', 1: 'B-mount', 2: 'I-mount'}

dir = "./model_save" #directory with presaved model and tokenizer settings

"""First of all, we need to load the pretrained parameters of model and tokenizer"""

tokenizer = BertTokenizer.from_pretrained(dir)
model = BertForTokenClassification.from_pretrained(dir,
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)

pipe = pipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
#aggregation strategy is defined because we did worpiece tokenization. So the output of the model originally is also pieces of the words and their labels. aggregation_strategy="simple" changes the output from worpieces to words
pipe("Ben Nevis is a popular hiking destination, with 150,000 people a year visiting the peak.")
