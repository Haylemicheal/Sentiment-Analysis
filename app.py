from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import requests
import csv

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
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
output_list = [["Text","Positve","Neutral","Negative"]]
for i in range(100):
    res = requests.get('https://api.pullpush.io/reddit/search/comment/?q=toyota&before='+str(before))
    for j in range(100):
        print("I am in "+ str(i) +" of"+ str(j))
        temp = []
        text = res.json()["data"][j]['body']
        # if (len(text)/4 > 514):
        #     text = text[0:514]
        temp.append(text)
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
        tokenizer.model_max_length = 512
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for x in range(scores.shape[0] -1, -1, -1):
            s = scores[ranking[x]]
            temp.append(str(s))
        output_list.append(temp)
        
    before = str(res.json()["data"][99]['created_utc'])
    
with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(output_list)