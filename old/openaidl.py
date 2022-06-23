# openaidl.py - For getting openai word embeddings
#               into a text file for notebook use.
import os
import openai
import time
from openai.embeddings_utils import get_embedding


openai.api_key = os.getenv('OPENAI_API_KEY')

embeds = []  # target word embeddings from model
vocab = []   # list of all our valid words


with open(u'./barlow/valid_vocab.txt', 'r') as f:
    for line in f:
        vocab.append(line.strip('\n'))

counter = 0
for word in vocab:
    em = get_embedding(word, engine="text-similarity-babbage-001")
    embeds.append(em)

    # this is an excessive amount of caution
    counter += 1
    if counter % 30 == 29:
        print(f"avoiding rate limit")
        time.sleep(120)


with open(u'./barlow/gpt_ada.txt', 'w') as f:   
    for em in embeds:
        for p in em:
            f.write(str(p) + ' ')
        f.write("\n")


print('done')
