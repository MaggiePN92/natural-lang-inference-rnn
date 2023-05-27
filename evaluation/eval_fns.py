import torch


@torch.no_grad()
def accuracy_packed_seqs(model, data_iter, device):
    model.eval()
    labels_true, predictions = [], []

    for premise, hypothesis, label_true, prem_len, hypo_len in data_iter:
        label_true = label_true.to(device)
        output = model(
            premise.to(device), hypothesis.to(device), prem_len, hypo_len
        )
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()

    return (torch.tensor(predictions) == torch.tensor(labels_true)).float().mean() * 100.0
