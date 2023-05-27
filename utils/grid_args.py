from dataclasses import dataclass


@dataclass
class GridArgs:
    def __init__(
        self,
        model_name,
        hidden_size,
        num_layers,
        dropout,
        lr,
        freeze,
        bidirectional,
        pooling,
        rnn_type,
        pooling_multiplier,
        epochs = 25,
        n_classes = 3,
        device = "cuda"
    ) -> None:
        """Holds arguments to use in grid search. """
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.freeze = freeze
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.device = device
        self.pooling = pooling
        self.rnn_type = rnn_type
        self.pooling_multiplier = pooling_multiplier
