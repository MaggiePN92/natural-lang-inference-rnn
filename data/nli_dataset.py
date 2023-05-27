import torch

class NLIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        premise, 
        hypothesis, 
        targets,
        embedding,
        data_prepper
    ) -> None:
        self.unk_idx = embedding.get_unk_idx()
        self.pad_idx = embedding.get_pad_idx()
        self.target_stoi = {
            'contradiction' : 1, 
            'entailment' : 2, 
            'neutral' : 0
        }
        self.targets = [self.target_stoi[t] for t in targets]
        self.embedding = embedding
        self.data_prepper = data_prepper
        self.hypothesis = self.tokenize(hypothesis)
        self.premise = self.tokenize(premise)

    def __getitem__(self, idx : int):
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        target = self.targets[idx]
        return torch.LongTensor(premise), torch.LongTensor(hypothesis), torch.LongTensor([target])

    def __len__(self):
        return len(self.targets)

    def tokenize(self, data):
        result = []

        for doc in data:
            prepped_txt = [
                self.embedding.get_index(token, default=self.unk_idx)
                for token in self.data_prepper(doc)
            ]
            if len(prepped_txt) == 0:
                prepped_txt = [self.pad_idx]

            result.append(prepped_txt)

        return result