from typing import List
from smart_open import open
import pandas as pd


def read_tsv_df(path : str) -> pd.DataFrame:
  d = [line.strip().split('\t') for line in open(path)]
  d2df = {
    k : [] for k in d[0]
  }

  for i,r in enumerate(d[1:]):
    
    try:
        d2df["label"].append(r[0])
        d2df["premise"].append(r[1])
        d2df["hypothesis"].append(r[2])
        d2df["tagged_premise"].append(r[2])
        d2df["tagged_hypothesis"].append(r[2])
    except:
      print(f"Error occoured at line: {i}. Skipping line.")
      break

  df = pd.DataFrame(data=d2df)
  return df


def read_tsv_nestedlist(path : str) -> List[List[str]]:
    """Reads any file and iterates each line in file. Splits
    each line on tab. 

    Args:
        path (str): path to .tsv or .tsv.gzip

    Returns:
        List[List[str]]: list where each row in dataset is a list (nested list).
    """
    # assumes first line is header 
    return [line.strip().split('\t') for line in open(path)][1:]


def read_tsv_lists(path : str):
  """Reads tsv/tsv.gzip file and makes lists for each
  columns. 

  Args:
      path (str): path to tsv file

  Returns:
      Tuple[List[str], ..., List[str]]: targets, premise, hypothesis, premise_pos, hypothesis_pos
  """
  # assumes first line is header 
  d = [line.strip().split('\t') for line in open(path)][1:]

  targets = []
  premise = []
  hypothesis = []
  premise_pos = []
  hypothesis_pos = []

  for ls in d:

      targets.append(ls[0])
      premise.append(ls[1])
      hypothesis.append(ls[2])
      premise_pos.append(ls[3])
      hypothesis_pos.append(ls[4])
  # returns five lists with labels and input
  return targets, premise, hypothesis, premise_pos, hypothesis_pos