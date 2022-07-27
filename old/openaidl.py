# openaidl.py - For getting openai word embeddings
#               into a text file for notebook use.
import os
import openai
import time
from tqdm import tqdm
from openai.embeddings_utils import get_embedding


openai.api_key = os.getenv('OPENAI_API_KEY')

embeds = []  # target word embeddings from model
vocab = []   # list of all our valid words

print('vocab time')
with open(u'../vocab/expanded_vocab.txt', 'r') as f:
    for line in f:
        vocab.append(line.strip('\n'))


with open(u'../gpt/gpt_ada.txt', 'w') as f:
    counter = 0
    for i in tqdm(range(len(vocab))):
        word = vocab[i]
        em = get_embedding(word, engine="text-similarity-ada-001")
        embeds.append(em)

        for p in em:
            f.write(str(p) + ' ')
        f.write("\n")

        # this is an excessive amount of caution
        counter += 1
        if counter % 30 == 29:
            print(f"avoiding rate limit")
            time.sleep(120)  
        
print('done')
