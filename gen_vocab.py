import unicodedata
import sys, os
from tqdm import tqdm
from collections import Counter
import pandas as pd
from pandarallel import pandarallel
tqdm.pandas()

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ' '.join(["".join(x) for x in output])

def replace_multi_whitespaces(line):
    return ' '.join(line.split())

def process_line(line):
    if line == '\n':
        return ""
    else:
        line = _run_split_on_punc(line)
        line = replace_multi_whitespaces(line)
        if line != "":
            return [tk for tk in line.split(" ") if tk.isalpha()]
        else:
            return ""

def proc_txt(file_path):	
    # with open(sys.argv[1], "r") as input_file:
    l_tokens = []
    input_file = open(file_path, "r")
    
    print(f"Load file '{file_path}'...")
    lines = input_file.readlines()
    for line in tqdm(lines):
        l_tokens.append(process_line(line))

    input_file.close()

    print("Generate vocabulary...")
    l_tokens = [tk for list_ in l_tokens for tk in list_]
    dic = Counter(l_tokens)
    dic_df = pd.DataFrame(dic.items(), columns=['word','freq'])

    dic_df.to_json(file_path.split(".")[0]+"_vocab.json")
    print("Done...")


def proc_df(folder_path, file_path):
    file_path = os.path.join(folder_path, file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    print(file_name)

    print("Load datafram...")
    data = pd.read_json(file_path)
    tokens = data.text.parallel_apply(lambda line: process_line(line))

    print("\nGenerate vocabulary...")
    tokens_list = tokens.to_list()
    tokens_list = [tk for list_ in tokens_list for tk in list_]
    dic = Counter(tokens_list)
    dic_df = pd.DataFrame(dic.items(), columns=['word','freq'])

    file_name = os.path.basename(file_path).split('.')[0]
    output_path = f"../chunks_vocab/{file_name}_"
    output_file = output_path+"vocab.json"
    dic_df.to_json(output_file)
    print(f"Done in '{output_file}'!")

def procFolder(folder_path):
    files = os.listdir(folder_path)
    for file in tqdm(files):
        proc_df(folder_path, file)

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True)
    # proc_df(sys.argv[1])

    folder = sys.argv[1]
    procFolder(folder)
