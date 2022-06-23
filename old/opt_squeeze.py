# opt_squeeze.py - A script for saving our common vocab as OPT-1.3b embeds
from transformers import GPT2Tokenizer, OPTModel
import torch
from tqdm import tqdm
import numpy as np

# This is how I'm getting the embedding from OPT.
def squeeze(word, tokenizer, model):
    # prepare inputs and model
    inputs = tokenizer(word, return_tensors="pt") # return pytorch tensors
    with torch.no_grad(): # added 6/16
        outputs = model(**inputs)

    # get embedding from final layer
    last_hidden_states = outputs.last_hidden_state # I have confirmed this is the same
                                                   # As the strategy from Tianyi

    # remove unnecessary dimension
    embeddings = torch.squeeze(last_hidden_states, dim=0)

    # convert to list
    # return [embed.tolist() for embed in embeddings][1]

    # add all of the vectors together
    final_embed = np.array(embeddings[1])
    if embeddings.shape[0] > 2:
        for next_emb in embeddings[2:]:
            final_embed += np.array(next_emb)
        final_embed /= embeddings.shape[0] - 1
    return final_embed

def main():
    print("starting")
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b", cache_dir="/scratch/mbarlow6/.cache")
    model = OPTModel.from_pretrained("facebook/opt-1.3b", cache_dir="/scratch/mbarlow6/.cache")

    path = u'/gpfs/fs1/home/mbarlow6/Desktop/Conceptual-Analysis/barlow/valid_vocab.txt'
    
    vocab = []
    embeds = []
    print("building vocab...")
    with open(path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
    print("done")

    print("getting embeds")
    for i in tqdm(range(len(vocab))):
        word = vocab[i]
        embeds.append(squeeze(word, tokenizer, model))

    print("saving")
    with open(u'./opt/1_3B.txt', 'w') as f:
        for em in embeds:
            for p in em:
                f.write(str(p) + ' ')
            f.write('\n')
    
    print("done")

if __name__ == "__main__":
    main()