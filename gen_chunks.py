# for f in *; do zip ${f%%.*}.zip $f; done

import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def splitDataframe(folder_path, file_path, megas=10):
    file_path = os.path.join(folder_path, file_path)
    file_size = int(os.stat(file_path).st_size/ (1024 * 1024))
    file_name = os.path.basename(file_path).split('.')[0]

    output_path = f"../chunks/{file_name}_"
    
    f = open(file_path, 'r')
    print("Generando pandas dataframe...")
    df = pd.DataFrame(f.readlines(), columns=['text'])
    f.close()

    print(f"Generando mini dataframes de {megas}Mb ...")
    t_lines = len(df)
    n_chunks = int(t_lines / (megas*t_lines/file_size))
    list_df = np.array_split(df, n_chunks)
    for ith, mini_df in enumerate(list_df):
        output_file = output_path+str(ith)+".json"
        mini_df.to_json(output_file)
    print(f"Done {ith} from '{file_name} {file_size}MB'!")

def procFolder(folder_path):
    files = os.listdir(folder_path)
    txt_files = [f for f in files if f.split('.')[1] == 'txt']
    for file in tqdm(txt_files):
        splitDataframe(folder_path, file)


if __name__ == '__main__':
    # file_name = sys.argv[1]
    # splitDataframe(file_name)
    folder = sys.argv[1]
    procFolder(folder)

