import pandas as pd
import os, sys
from tqdm import tqdm
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True)


def genDics(folder):
    print("Generates dics from .json files...")
    name2key = lambda filename: filename.split("_")[1]+"_"+filename.split("_")[3].split(".")[0]
    vocab_files = os.listdir(folder)
    vocab_files = [(file, name2key(file)) for file in vocab_files]
    # vocab_files = [(file, file.split("_")[1]) for file in vocab_files]

    dics_list = []
    for file_name,sigla in tqdm(vocab_files):
        df = pd.read_json(os.path.join(folder,file_name))
        dic = df.set_index('word').to_dict()['freq']
        dics_list.append((dic,sigla))
    return dics_list

def genVocabs(dics_list):
    print("Generates full vocab ...")
    vocab_dic = dics_list[0][0].copy()
    for dic,_ in tqdm(dics_list[1:]):
        for word,freq in dic.items():
            v = vocab_dic.get(word)
            if v:
                vocab_dic[word] = freq+v
            else:
                _ = vocab_dic.setdefault(word,freq)
    return vocab_dic

def addPercentiles(full_dict):
    print("Generate vocab as datafrme format and calculate perentiles...")
    full_vocab = pd.DataFrame(full_dict.items(), columns=['word','freq'])
    full_vocab = full_vocab.sort_values(by=['freq'], ascending=False)

    # Remove top 100 words and word with frequency less than 4
    full_vocab = full_vocab[100:]
    full_vocab = full_vocab[full_vocab.freq > 3] 

    # Calculate acumulative frequency
    only_freq = pd.DataFrame(full_vocab.freq.unique(),columns=['freq'])
    only_freq['cum_freq'] = only_freq.loc[::-1, 'freq'].cumsum()[::-1]
    full_vocab = pd.merge(full_vocab,only_freq, on='freq', how='left')

    # Calculate percentile
    max_cum_freq = full_vocab.cum_freq[0]
    full_vocab['percentile'] = full_vocab.cum_freq.parallel_apply(lambda cf: round(cf/max_cum_freq,5))

    return full_vocab

def assignSplitSubsets(full_vocab, dics_list):
    print("Assign split_subsets and its freq")
    def assign(word):
        subset = {}
        for dic,sigla in dics_list:
            f1 = dic.get(word)
            if f1:
                f2 = subset.get(sigla)
                if f2:
                    subset[sigla] = f2 + f1
                else:
                    _ = subset.setdefault(sigla,f1)
        return [list(subset.keys()), list(subset.values())]


    # temp = full_vocab.word.parallel_apply(lambda w: assign(w))
    temp = full_vocab.word.progress_apply(lambda w: assign(w))
    full_vocab[['in_subset','in_subset_freq']] = temp.to_list()
    return full_vocab


def assignDicSubsets(full_vocab, dics_list):
    print("Assign subsets and its freq")
    def assign(word):
        subset = {}
        for dic,sigla in dics_list:
            sigla = sigla.split("_")[0]
            f1 = dic.get(word)
            if f1:
                f2 = subset.get(sigla)
                if f2:
                    subset[sigla] = f2 + f1
                else:
                    _ = subset.setdefault(sigla,f1)
        return subset

    # temp = full_vocab.word.parallel_apply(lambda w: assign(w))
    temp = full_vocab.word.progress_apply(lambda w: assign(w))
    full_vocab['subset_dic'] = temp.to_list()
    return full_vocab


    
if __name__ == '__main__':
    folder = sys.argv[1]

    dict_list = genDics(folder)
    full_dict = genVocabs(dict_list)
    fb = addPercentiles(full_dict)
    fb = assignSplitSubsets(fb, dict_list)
    fb_w_dic = assignDicSubsets(fb, dict_list)


    # Assign subsets to each word
    file_out = "full_vocab_v3.json"
    fb_w_dic.to_json(file_out)
    print(f"Done in '{file_out}'!")