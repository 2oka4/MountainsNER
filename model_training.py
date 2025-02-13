# -*- coding: utf-8 -*-

from transformers import TrainingArguments, Trainer, BertTokenizer, BertForTokenClassification
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

train_df = pd.read_csv("mountains_preprocessed.csv") #loading preprocessed set
label_df = pd.read_csv("mountains_labeled.csv") #the only reason we load this csv is to create label2id dictionary

train_df.shape #there are 404 sentences in the training set

train_df

label2id = {k: v for v, k in enumerate(label_df.Tag.unique())} #tags to indices (we will need this dictionary as model works with indices, not with tags)
id2label = {v: k for v, k in enumerate(label_df.Tag.unique())} #indices to tags
label2id

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #tokenizer

max_length=128 #the maximum number of labels (tokens) the tokenizer will output

"""Bert relies on wordpiece tokenization. It means that we need to define labels at the wordpiece-level, rather than word-level."""

def tokenization(sentence, text_labels, tokenizer):

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

"""Next, we define a regular PyTorch dataset class (which transforms examples of a dataframe to PyTorch tensors). Here, each sentence gets tokenized, the special tokens that BERT expects are added, the tokens are padded or truncated based on the max length of the model, the attention mask is created and the labels are created based on the dictionary which we defined above"""

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]
        word_tags = self.data.word_tags[index]
        tokenized_sentence, labels = tokenization(sentence, word_tags, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [label2id[label] for label in labels]

        return {
              'input_ids': torch.tensor(ids, dtype=torch.long),
              'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
              'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

"""Let`s create validation set"""

train_size = 0.88
train_dataset = train_df.sample(frac=train_size,random_state=42)
val_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

training_set = dataset(train_dataset, tokenizer, max_length)
val_set = dataset(val_dataset, tokenizer, max_length)

training_set[0] #structure is appropriate

val_set[0]

"""Here we define the model, BertForTokenClassification, and load it with the pretrained weights of "bert-base-uncased"."""

model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)

"""Also we define metrics function the will calculate precision, recall, f1 score, accuracy."""

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    # Extract predictions and true labels
    predictions = pred.predictions
    labels = pred.label_ids
    print("Predictions shape:", predictions.shape)
    print("Labels shape:", labels.shape)
    # For token classification, predictions are the argmax of logits
    predicted_labels = predictions.argmax(axis=-1)

    # Flatten arrays and filter out special tokens (0 in labels)
    true_labels = []
    pred_labels = []
    for true, pred in zip(labels, predicted_labels):
        for t, p in zip(true, pred):
            if t != 0 :  # Ignore special tokens
                true_labels.append(t)
                pred_labels.append(p)
    print("True labels (sample):", true_labels[:10])
    print("Predicted labels (sample):", pred_labels[:10])

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )
    accuracy = accuracy_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

"""We will use Hugging Face Trainer class for training in PyTorch. First of all we need to define training hyperparameters"""

arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    learning_rate=1e-4,
    save_total_limit=3,
    save_strategy='steps',
    eval_strategy="steps",
    load_best_model_at_end=True,
    report_to = "none"
    )

import transformers
transformers.logging.set_verbosity_info() #we use it to show some addtional information during training process

trainer = Trainer( #define Trainer
      model=model,
      args=arguments,
      train_dataset=training_set,
      eval_dataset=val_set,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics)
trainer.train()

"""Evaluating our model on the validation set"""

trainer.evaluate()

output_dir = "./model_save"

model.save_pretrained(output_dir) #saving model configs (including weights)

tokenizer.save_pretrained(output_dir)
