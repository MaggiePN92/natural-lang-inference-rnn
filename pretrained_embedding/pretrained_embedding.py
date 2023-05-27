import torch
from pretrained_embedding.load_embedding import load_embedding
import gensim


def get_embedding(args):
  if args.fasttext:
    model = gensim.models.fasttext.load_facebook_vectors(args.path2embedding) 
    return FastTextExtension(model)
  else:
    return PretrainedEmbedding(args.path2embedding)
  

class PretrainedEmbedding:
  def __init__(
    self, 
    path : str,
    fasttext = False
  ) -> None:
    """Loads and implements some helper methods for gensim word
    embeddings. 

    Args:
        path (str): path to embedding that should be loaded. 
    """
    if fasttext:
      self.emb_model = gensim.models.fasttext.load_facebook_vectors(path)
    else:
      self.emb_model = load_embedding(path)

    print(f"Embedding loaded with vector size = {self.emb_model.vector_size}.")

    self.emb_model["[UNK]"] = torch.tensor(
      self.emb_model.vectors
    ).mean(dim = 0).numpy()

    self.emb_model["[PAD]"] = torch.zeros(
      self.emb_model.vector_size
    ).numpy()

  def get_index(self, token, default):
    return self.emb_model.get_index(token, default=default)

  def get_pad_idx(self):
    return self.emb_model.get_index("[PAD]")
  
  def get_unk_idx(self):
    return self.emb_model.get_index("[UNK]")

  def get_vectors(self):
    return torch.from_numpy(self.emb_model.vectors)


class FastTextExtension:
  def __init__(self, emb_model):
    """THIS IS NOT USED - IS NOT CORRECTLY IMPLEMENTED"""
    self.emb_model = emb_model
    self.max_idx = max(self.emb_model.key_to_index.values())
    self.key_to_index = {
        "[PAD]" : max(self.emb_model.key_to_index.values()) + 1
    }
    self.vec = torch.zeros((1, emb_model.vector_size))
    
  def add_token_to_vec(self, token):
    self.key_to_index[token] = self.max_idx + len(self.key_to_index.values())
    self.vec = torch.cat(
        (self.vec, torch.from_numpy(self.emb_model[token]).unsqueeze(0)), 
        dim=0
    )

  def get_pad_idx(self):
    return self.key_to_index["[PAD]"]
  
  def get_unk_idx(self):
    # There should be no unknown tokens when using FastText
    return None
    
  def get_index(self, token, **kwargs):
    try:
      return self.emb_model.key_to_index[token]
    except KeyError:
      try:
        return self.key_to_index[token]
      except KeyError:
        self.add_token_to_vec(token)
        return self.key_to_index[token]

  def get_vectors(self):
    vecs = torch.from_numpy(self.emb_model.vectors)
    return torch.concat((vecs, self.vec), dim=0)
