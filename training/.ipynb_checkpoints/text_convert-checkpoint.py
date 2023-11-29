import numpy as np
import multiprocessing as mp
import torch

import os
from preprocessing import preprocess,  metrics, w2v
remap = {'a':'an', 'and':'also', 'of':'in', 'to':'at'}

base_dir = '../'
def run(filepath, outpath):
    # read the raw text file
    print('opening the file:', filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except:
        with open(filepath, 'r', encoding='cp437') as file:
            text = file.read()
    
    # extract metrics then do preprocessing
    met = metrics(text)
    
    # do processing 
    proc = preprocess(text)
    processed_text_words = proc.split()# split by whitespace
    essay_words = [remap[word] if word in remap.keys() else word for word in processed_text_words]
    
    # do word2vec
    vecs = []
    missing_words =[]
    for word in essay_words:
        try:
            vec = w2v.get_vector(word)
            vecs.append(vec)
        except KeyError:
            # this means that the word isn't in the w2v
            missing_words.append(word)
    vecs = np.array(vecs)
    print('NP array shape:', vecs.shape)

    #print all the words that are missing
    unique_missing =  " ".join(list(np.unique(np.array(missing_words))))
    print('missing: ',unique_missing)
    
    # write to the outfile
    metric_cols = ('"%s",' * len(met))%tuple(met)
    write_out = '"%s","%s","%s",'%(filepath," ".join(essay_words), unique_missing) + metric_cols + '\n'
    with open(outpath, 'a') as f:
        f.write(write_out)
        
    return torch.from_numpy(vecs).reshape(-1,300)

if __name__=='__main__': # took about an hour to run on all the human training essays
    texts_dir = base_dir+'data/GPT_essays/'
    out_file = base_dir+'data/train_GPT_metrics.csv'
    fnames = os.listdir(texts_dir)
    
    
    with open(out_file, 'w') as f:
        f.write('"file","processed_text","missing_words","met1","met2","met3","met4","met5",\n')
    
    
    Tensor_list = []
    for fname in fnames:
        tensor_form = run(texts_dir+fname, out_file)
        Tensor_list.append(tensor_form)
    
    
    torch.save(Tensor_list, base_dir+'data/train_GPT_tensor.pt')
    print('Done!')