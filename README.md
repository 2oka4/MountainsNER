# Natural Language Processing: Named Entity Recognition

## Overview
The goal of this project is to create a named entity recognition (NER) model for identifying mountain names in text. For this task, the pretrained BERT language model (LLM) was used.

## 1. Data Preprocessing
### Dataset
The training dataset is located in the *data* folder. The *Mountains_labeled.csv* dataset was generated using ChatGPT. It contains three columns:
1. **Word**: Individual words from the text.
2. **Tag**: Inside–outside–beginning (IOB) tagging for each word.
3. **Sentence Number**: The sentence number to which the word belongs.

Although the dataset includes IOB tagging, some preprocessing steps were required. These steps were implemented in the *Dataset_creation.ipynb* file. After preprocessing, the dataset was saved as *mountains_preprocessed.csv*, which is also available in the *data* folder.

## 2. Model Training
### Process
The complete model training process is described in the *model_training.py* file. The steps include:

1. **Tokenization**: The *mountains_preprocessed.csv* dataset is tokenized and transformed into a PyTorch dataset. 
   - Tokenization is essential for BERT as it converts raw text into numerical tokens (subwords in this case) that the model can process.
   - Special tokens (`[CLS]` and `[SEP]`) are added to distinguish between sentences, and padding is applied to ensure uniform input size.
   - Attention masks are generated to distinguish real tokens from padding, ensuring that padding is ignored during computations.

2. **Metrics Definition**: A `compute_metrics` function is defined to evaluate model performance.

3. **Model Definition**: The *BertForTokenClassification* class, a pretrained BERT model, is used for token classification.

4. **Training**: The model is trained using the *Trainer* class from Hugging Face, which simplifies the training process.

5. **Saving the Model**: After training, the model's parameters (weights, etc.) are saved in the *model_save* folder. These files can be downloaded from Google Drive using the link provided in the *model_save.txt* file.

## 3. Model Inference
### Process
Model inference was performed on another dataset, *mountain_sentences.csv*, located in the *Data_used_for_predictions* folder. The inference process is implemented in the *model_inference.py* file. 

Steps:
1. The tokenizer and model, along with the parameters saved during training, are loaded from the *model_save* folder.
2. Inference is performed using the Hugging Face *pipeline* class, which automates all data transformation steps.

The *mountain_sentences.csv* dataset contains one sentence per row. Predictions for each sentence are added as a new column in the resulting *predictions.csv* file, available in the *Data_used_for_predictions* folder.

Additionally, the *model_inference_with_edge_cases.ipynb* notebook contains the same inference process but also includes edge case handling and analysis.

## Additional Information
- **potential_improvements.pdf**: Outlines potential steps to improve model performance.
- **requirements.txt**: Lists all libraries used in the project.
