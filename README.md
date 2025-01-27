# Natural Language Processing. Named entity recognition
## The objective of this project is creating a named entity recognition (NER) model for the identification of mountain names inside the texts. For this task the pretrained BERT LLM model was used.
## 1. Data Preprocessing
### The training dataset can be found in *data* folder. *Mountains_labeled.csv* dataset was generated by using ChatGPT. First column is word, second is its tag, third is sentence number of the word.  NER model requires special Inside–outside–beginning (IOB) tagging for every word, so second Tag column holds this tagging. However, this dataset still need some preprocessing steps which where taken in the *Dataset_creation.ipynb*. After taking these steps, *mountains_preprocessed.csv* was created. This file can also be found in *data* folder.
## 2. Model training
### The whole training process can be found in *model_training.py* file. First of all, we need to use tokenizer to our *mountains_preprocessed.csv* and transform it to Pytorch dataset. A tokenizer is essential for BERT because it transforms raw text into a format that the model can process. BERT expects input data to be numeric, represented by tokens (words or subwords, in our case subwords). Additionaly, tokenizer adds special tokens ([CLS], [SEP]) called '*pads*' to distinguish between sentences. Also, The tokenizer generates '*attention masks*' to distinguish real tokens from padding. This ensures that padding tokens are ignored during computations. After transforming our dataset to Pytorch dataset we define *compute_metrics* function to evaluate our model. As a model we have used *BertForTokenClassification* pretrained BERT class. For training the model we use another convenient tool called *Trainer* which is also provided by Hugging Face. Аfter training the model we save its paramaters (weights, etc) in model_save folder. These files can be obtained from the Google Driveб link to which is in *model_save.txt* file. 
## 3. Model inference
### Model inference was done on another generated dataset *mountain_sentences.csv* which can be found in *Data_used_for_predictions* folder. The inference process is written in *model_inference.py* file. To begin with, we define our tokenizer and model with parameters obtained from model training process (from *model_save* folder). To tackle the process of inference easily we use one more Hugging Face class called pipeline which does all of the data transformation steps automatically. *mountain_sentences.csv* contains one sentence per row and so the prediction (for each sentence) is written in the next column. Updated dataset (with mountain names that were (or not) found in each sentence) is called *predictions.csv* is also in *Data_used_for_predictions* folder. Addtionaly, *model_inference_with_edge_cases.ipynb* contains the same model inference steps as well as going through some edge cases.
## *potential_improvements.pdf* contains some of possible steps tha can be taken in order to improve a model perfomance. *requirements.txt* holds the names of all libraries used in the project.
