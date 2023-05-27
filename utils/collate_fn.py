import torch


class CollateFunc:
    """Not in use."""
    def __init__(self, padding_idx, max_len) -> None:
        self.padding_idx = padding_idx
        self.max_len = max_len

    def __call__(self, samples):
        premise_ids, hypothesis_ids, targets = zip(*samples)

        premise_padded = torch.nn.utils.rnn.pad_sequence(
            premise_ids,
            batch_first=True,
            padding_value=self.padding_idx
        )

        hypothesis_padded = torch.nn.utils.rnn.pad_sequence(
            hypothesis_ids,
            batch_first=True,
            padding_value=self.padding_idx
        )
        
        premise_padded = premise_padded[:,:self.max_len]
        hypothesis_padded = hypothesis_padded[:,:self.max_len]
        targets_pt = torch.LongTensor(targets)

        return premise_padded, hypothesis_padded, targets_pt


class CollateFuncPacking:
    def __init__(self, max_len : int, pad_idx : int) -> None:
        """Collate function to use when sequences are to be Packed
        later on. 

        Args:
            max_len (int): maximum length of sequences.
            pad_idx (int): index of padding in embedding. 
        """
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __call__(self, samples):
        """Makes CollateFuncPacking callable. 

        Args:
            samples : Premise, hypothesis and targets gathered 
            from dataset. 

        Returns:
            PaddedSequnces, PaddedSequnces, Tensor, List[int], List[int]
        """
        # gather ids from premise and hypothesis and slice them to reduce
        # maximum sequence lengths 
        premise_ids = [p[:self.max_len] for p, _, _ in samples]
        hypothesis_ids = [h[:self.max_len] for _, h, _ in samples]
        # measure lengths of sequences, needed for padding
        hypothesis_lens = [len(x) for x in hypothesis_ids]
        premise_lens = [len(x) for x in premise_ids]
        
        targets = [t for _, _, t in samples]
        # pad premise and hypothesis 
        premise_ids_padded = torch.nn.utils.rnn.pad_sequence(
            premise_ids, batch_first=True, padding_value=self.pad_idx
        )

        hypothesis_ids_padded = torch.nn.utils.rnn.pad_sequence(
            hypothesis_ids, batch_first=True, padding_value=self.pad_idx
        )
        
        targets_pt = torch.LongTensor(targets)

        return premise_ids_padded, hypothesis_ids_padded, targets_pt, premise_lens, hypothesis_lens
