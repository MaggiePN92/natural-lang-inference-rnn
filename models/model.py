import torch


class ANN(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_size, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout)
        )

    def forward(self, x : torch.Tensor):
        x = self.layer(x)
        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        embedding_model,
        pooling,
        args,
    ):
        super().__init__()
        # padding idx to know what to mask
        self.pad_idx = embedding_model.get_pad_idx()
        # get vectors from embedding model
        embedding_weights = embedding_model.get_vectors()

        # embedding -> dropout -> linear
        # output dim = [batch size, sequence length, hidden size]
        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(
                embedding_weights, freeze=args.freeze
            ),
            torch.nn.Dropout(args.dropout),
        )
        # projection layer
        self.first_linear = torch.nn.Linear(
            embedding_model.emb_model.vector_size * 2,
            args.hidden_size
        )
        # list of linear layers
        self.layers = torch.nn.ModuleList([
           ANN(args) for _ in range(args.num_layers)
        ])
        # init pooling
        self.pooling = pooling(args)
        # project into n_classes
        self.output = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_size, args.n_classes)
        )
      
    def forward(
        self, premise_ids, hypothesis_ids, *args
    ):
        # make masks
        premise_mask = (premise_ids == self.pad_idx)
        hypothesis_mask = (hypothesis_ids == self.pad_idx)
        # extract word embeddings
        premise_emb = self.embedding(premise_ids)
        hypothesis_emb = self.embedding(hypothesis_ids)
        # from sequence to fix sized representation
        premise = self.pooling(premise_emb, premise_mask)
        hypothesis = self.pooling(hypothesis_emb, hypothesis_mask)
        # concat the fixed size repr. 
        x = torch.concat((premise, hypothesis), dim=1)
        # project x into hidden size
        x = self.first_linear(x)
        # linear with residual connections
        for layer in self.layers:
            x = layer(x) + x

        logits = self.output(x)
        return logits
