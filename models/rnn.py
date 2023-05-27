from torch import nn
import torch


class RNN(torch.nn.Module):
    def __init__(self, args, embedding_dim) -> None:
        """RNN that implements one RNN for premise and one for hypothesis."""
        super().__init__()
        self.rnn_type = args.rnn_type.lower()
        self.recurrent_module = {
            'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.rnn_type]

        self.args = args

        self.rnn_premise = self.recurrent_module(
            embedding_dim,
            args.hidden_size, 
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_first = True,
            bidirectional = args.bidirectional
        )

        self.rnn_hypothesis = self.recurrent_module(
            embedding_dim,
            args.hidden_size, 
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_first = True,
            bidirectional = args.bidirectional
        )

    def init_states(self):
        """Method for init cell and hidden state"""
        hc_dim = 2 * self.args.num_layers if self.args.bidirectional else self.args.num_layers
        h_0 = nn.Parameter(
            torch.zeros((hc_dim, 1, self.args.hidden_size))
        )
        c_0 = nn.Parameter(
            torch.zeros((hc_dim, 1, self.args.hidden_size))
        )
        return h_0, c_0

    def forward(
            self, premise_packed, hypothesis_packed, c_0, h_0
    ):
        if self.rnn_type=="lstm":
            x_prem, (h_n_prem, c_n_pre,) = self.rnn_premise(premise_packed, (c_0, h_0))
            x_hypo, (h_n_hypo, c_n_hypo) = self.rnn_hypothesis(hypothesis_packed, (c_0, h_n_prem))
        else:
            x_prem, h_n_prem = self.rnn_premise(premise_packed, h_0)
            x_hypo, h_n_hypo = self.rnn_hypothesis(hypothesis_packed, h_n_prem)

        return x_prem, h_n_prem, x_hypo, h_n_hypo


class Model(torch.nn.Module):
    def __init__(
        self,
        embedding_model,
        pooling,
        args,
    ):
        super().__init__()
        self.pad_idx = embedding_model.get_pad_idx()
        self.device = args.device

        embedding_weights = embedding_model.get_vectors()
        # embedding -> dropout -> linear
        # output dim = [batch size, sequence length, hidden size]
        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(
                embedding_weights, freeze=args.freeze
            ),
            torch.nn.Dropout(args.dropout),
        )

        self.rnn = RNN(args, embedding_model.emb_model.vector_size)
        # fixed size repr. depends on what pooling mechanism is used
        self.pooling = pooling(args)
        input_dim_linear = (args.hidden_size * 2 if args.bidirectional \
            else args.hidden_size) * args.pooling_multiplier
        # last layer of RNN
        self.output = torch.nn.Linear(
            input_dim_linear, args.n_classes
        )
      
    def forward(
        self, premise_ids, hypo_ids, 
        premise_lens, hypothesis_lens
    ):
        # B : Batch size
        B = premise_ids.size(0)
        # construct masks for premise and hypothesis
        premise_mask = (premise_ids == self.pad_idx).to(self.device)
        hypothesis_mask = (hypo_ids == self.pad_idx).to(self.device)
        # extract embeddings for each
        prem_emb = self.embedding(premise_ids)
        hypo_emb = self.embedding(hypo_ids)
        # pack sequences for more efficient compute
        premise_packed = torch.nn.utils.rnn.pack_padded_sequence(
            prem_emb, premise_lens, batch_first=True, enforce_sorted=False)
        hypothesis_packed = torch.nn.utils.rnn.pack_padded_sequence(
            hypo_emb, hypothesis_lens, batch_first=True, enforce_sorted=False)
        # init states - they are learnable
        c_0, h_0 = self.rnn.init_states()
        # expand cell and hidden init states 
        c_0, h_0 = c_0.expand(-1, B, -1).to(self.device), h_0.expand(-1, B, -1).to(self.device)
        # pass packed seqs to rnn
        packed_prem_outp, h_n_prem, packed_hypo_outp, h_n_hypo = self.rnn(
            premise_packed, hypothesis_packed, c_0, h_0
        )
        # pad sequnces
        padded_prem_outp, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_prem_outp, batch_first=True
        )
        padded_hypo_outp, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_hypo_outp, batch_first=True
        )
        # make fix sized representation of padded seqs
        fixed_size_repr = self.pooling(
            padded_prem_outp, h_n_prem, 
            padded_hypo_outp, h_n_hypo,
            premise_mask, hypothesis_mask
        )
        # generate logits
        logits = self.output(fixed_size_repr)
        return logits
