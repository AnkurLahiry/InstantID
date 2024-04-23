from detoxify import Detoxify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bertSummarizer import *
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def is_toxic_prompt(prompt):
    # Check whether the prompt is a toxic or not
    # We are using Detoxify library to check the prompt
    # There are six classes - toxicity, severe_toxicity, obscene, threat, insult, identity_attack
    # However we don't need to check the class, we need to know if any parameter is over a given user defined threshold.
    # We have defined toxic threshold as 0.20, it may be fined tunned.
    # return value is a boolean if a prompt is toxic or not. 
    results = Detoxify('original').predict(prompt)
    toxic_threshold = 0.20
    print(results)
    #detects 6 classes
    #toxicity, severe_toxicity, obscene, threat, insult, identity_attack
    # print(results)
    # if results['toxicity'] > toxic_threshold:
    #     return True 
    # if results['severe_toxicity'] > toxic_threshold:
    #     return True 
    # if results['obscene'] > toxic_threshold:
    #     return True 
    # if results['threat'] > toxic_threshold:
    #     return True
    # if results['insult'] > toxic_threshold:
    #     return True 
    # if results['identity_attack'] > toxic_threshold:
    #     return True 
    # return False

    for key, value in results.items():
        if value > toxic_threshold:
            return True 
    return False
    

def is_phising_prompt(prompt):
    #Check whether a prompt is phishing or not
    #Or it may contain false information.
    #The motivation of this function is the given prompt has any information that is false. 
    #As we are using softmax, the generic threshold is 0.5 for binary classification. 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    # Forward pass, no gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)
        prediction_scores = softmax(outputs.logits, dim=1)
    
    # Get the probability of the prompt being harmful
    print(prediction_scores)
    harmful_prob = prediction_scores[0][1].item()  # Assuming index 1 is for 'harmful'
    threshold = 0.5
    print(harmful_prob)
    if harmful_prob > threshold:
        return True
    else:
        return False
    

def t5_summary_generation(prompt):
    #The function takes input a long prompt, and returns the summary of the prompt to faster the image generation process
    #We use t5-large model to summarize the prompt. 
    #return value is summerized prompt of a given prompt if it is above a max_length. 
    #We assume that for a certain length we don't want to make summary
    max_length = 512
    if len(prompt) < max_length + 1: 
        return prompt
    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def bert_summary(prompt):
    #The function takes input a long prompt, and returns the summary of the prompt to faster the image generation process
    #We use bert model to summarize the prompt. 
    #return value is summerized prompt of a given prompt if it is above a max_length. 
    #We assume that for a certain length we don't want to make summary
    max_length = 512
    if len(prompt) < max_length + 1:
        return prompt
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentences = nltk.tokenize.sent_tokenize(prompt)
    tokenized_sentences = [tokenizer.encode(s, add_special_tokens=True, max_length=max_length, truncation=True) for s in sentences]
    max_len = max([len(s) for s in tokenized_sentences])
    padded = [s + [0] * (max_len - len(s)) for s in tokenized_sentences]
    attention_mask = [[float(i > 0) for i in seq] for seq in padded]
    input_ids, attention_mask = torch.tensor(padded), torch.tensor(attention_mask)
    model = BertSummarizer()
    with torch.no_grad():
        probs = model(input_ids, attention_mask=attention_mask)
    threshold = 0.5  # You can adjust this threshold
    summary = []
    
    # Assume probs is of shape [number of sentences, tokens per sentence]
    # We take the mean or max of probabilities along the token dimension
    sentence_probs = torch.max(probs, dim=1).values  # or torch.mean(probs, dim=1) if averaging is preferred

    for idx, sentence_prob in enumerate(sentence_probs):
        if sentence_prob.item() > threshold:  # Convert tensor to Python scalar with .item()
            sentence = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
            summary.append(sentence)
    
    return ' '.join(summary)


nltk.download('punkt')
nltk.download('stopwords')

def keyword_extractor(prompt):
    #The function takes input a long prompt, and returns the keyword of the prompt to faster the image generation process
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(prompt)
    filtered_words = [word for word in words if word not in stop_words and word.isalnum()]

    # Calculate frequency distribution
    freq_dist = nltk.FreqDist(filtered_words)
    keywords = [word for word, freq in freq_dist.items() if freq > 1]  # Adjust threshold as needed
    
    return ', '.join(keywords)  # Join keywords as a comma-separated string




