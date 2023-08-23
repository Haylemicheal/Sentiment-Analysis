from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import requests
import csv

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

before = 1680296760

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

output_list = [["Text", "Positive", "Neutral", "Negative"]]
for i in range(100):
    res = requests.get('https://api.pullpush.io/reddit/search/comment/?q=toyota&before=' + str(before))
    for j in range(100):
        print("I am in " + str(i) + " of " + str(j))
        temp = []
        text = res.json()["data"][j]['body']
        temp.append(text)
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
        tokenizer.model_max_length = 512
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()  # Use logits instead of [0][0]
        scores = softmax(scores)

        for s in scores:
            temp.append(str(s))
        output_list.append(temp)
        
    before = str(res.json()["data"][99]['created_utc'])

# Use text mode ("w") instead of binary mode ("wb") for writing
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
