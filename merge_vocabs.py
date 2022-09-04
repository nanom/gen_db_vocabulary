import pandas as pd
import os, sys
from tqdm import tqdm
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True)


def genDics(folder):
    print("Generates dics from .json files...")
    vocab_files = os.listdir(folder)
    vocab_files = [(file, file.split("_")[1]) for file in vocab_files]

    dics_list = []
    for file_name,sigla in tqdm(vocab_files):
        df = pd.read_json(os.path.join(folder,file_name))
        dic = df.set_index('word').to_dict()['freq']
        dics_list.append((dic,sigla))
    return dics_list

def genVocabs(dics_list):
    print("Generates full vocab ...")
    vocab_dic = dics_list[0].copy()
    for dic,_ in dics_list[1:]:
        for word,freq in tqdm(dic.items()):
            v = vocab_dic.get(word)
            if v:
                vocab_dic[word] = freq+v
            else:
                _ = vocab_dic.setdefault(word,freq)
    return vocab_dic

def assignSubsets(full_vocab, dics_list):
    def assign(word):
        subset = []
        for dic,sigla in dics_list:
            v = dic.get(word)
            if v:
                subset.append(sigla)
        return list(set(subset))

    full_vocab['in_subset'] = full_vocab.text.parallel_apply(lambda w: assign(w))

    return full_vocab

def addPercentiles(vocab_dic):
    print("Generate and save vocab as datafrme format ...")
    full_vocab = pd.DataFrame(vocab_dic.items(), columns=['word','freq'])
    full_vocab = full_vocab.sort_values(by=['freq'], ascending=False)

    # Calculate acumulative frequency
    only_freq = pd.DataFrame(full_vocab.freq.unique(),columns=['freq'])
    only_freq['cum_freq'] = only_freq.loc[::-1, 'freq'].cumsum()[::-1]
    full_vocab = pd.merge(full_vocab,only_freq, on='freq', how='left')

    # Calculate percentile
    max_cum_freq = full_vocab.cum_frec[0]
    full_vocab['percentile'] = full_vocab.cum_freq.parallel_apply(lambda cf: cf/max_cum_freq)

    return full_vocab

    
if __name__ == '__main__':
    folder = sys.argv[1]

    dics_list = genDics(folder)
    vocab_dic = genVocabs(dics_list)
    full_vocab = addPercentiles(vocab_dic)
    full_vocab = assignSubsets(full_vocab, dics_list)

    # Assign subsets to each word
    file_out = "full_vocab.json"
    full_vocab.to_json(file_out)
    print(f"Done in '{file_out}'!")
