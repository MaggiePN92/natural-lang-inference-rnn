from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import string
from typing import List


class DataPrep:
  def __init__(self) -> None:
    # remove ' from string with punctuations marks, other punctuation marks will
    # remain: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    self.PUNCTS = string.punctuation.replace("'", "")

  def __call__(self, txt : str) -> List[str]:
    # text is casefolded, chars in PUNCTS are removed and double white spaces are reduced
    # to single white space
    txt = txt.lower().translate(str.maketrans(self.PUNCTS, ' '*len(self.PUNCTS)))
    txt = [t.strip() for t in txt.split()]
    return txt

class DataPrepPos(DataPrep):
  def __init__(self) -> None:
    super().__init__()
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    self.PUNCTS = self.PUNCTS.replace("_", "")
    self.lemmatizer = WordNetLemmatizer()

  def __call__(self, txt : str) -> List[str]:
    processed_sent = []

    txt = txt.translate(str.maketrans(self.PUNCTS, ' '*len(self.PUNCTS)))

    for token in txt.split():
      
      if token.endswith("_PUNCT"):
        continue
      
      pos_start = token.find("_")
      # lemmatize tokens, probably not necesary as tokens are POS tagged
      if token.endswith("_NOUN"):
        processed_sent.append(
          f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.NOUN).lower()}{token[pos_start:]}"
        )
      elif token.endswith("_VERB"):
        processed_sent.append(
          f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.VERB).lower()}{token[pos_start:]}"
        )
      elif token.endswith("_ADJ"):
        processed_sent.append(
          f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.ADJ).lower()}{token[pos_start:]}"
        )
      elif token.endswith("_ADV"):
        processed_sent.append(
          f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.ADV).lower()}{token[pos_start:]}"
        )
      else:
        processed_sent.append(
            f"{token[:pos_start].lower()}{token[pos_start:]}")
      
    return processed_sent


class DataPrepLemma(DataPrep):
    def __init__(self) -> None:
      super().__init__()
      nltk.download('wordnet')
      nltk.download('omw-1.4')
      self.PUNCTS = self.PUNCTS.replace("_", "")
      self.lemmatizer = WordNetLemmatizer()

    def __call__(self, txt : str) -> List[str]:
      processed_sent = []

      txt = txt.translate(str.maketrans(self.PUNCTS, ' '*len(self.PUNCTS)))

      for token in txt.split():
        
        if token.endswith("_PUNCT"):
          continue
        
        pos_start = token.find("_")
        # lemmatize, case fold and remove pos tag 
        if token.endswith("_NOUN"):
          processed_sent.append(
            f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.NOUN).lower()}"
          )
        elif token.endswith("_VERB"):
          processed_sent.append(
            f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.VERB).lower()}"
          )
        elif token.endswith("_ADJ"):
          processed_sent.append(
            f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.ADJ).lower()}"
          )
        elif token.endswith("_ADV"):
          processed_sent.append(
            f"{self.lemmatizer.lemmatize(token[:pos_start], wordnet.ADV).lower()}"
          )
        else:
          processed_sent.append(
              f"{token[:pos_start].lower()}")
        
      return processed_sent
