from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import data.data_prep as data_prep
from utils.collate_fn import CollateFuncPacking
from data.nli_dataset import NLIDataset
import torch
from pretrained_embedding.pretrained_embedding import PretrainedEmbedding
from models.rnn import Model
from models.lstm_pooling import MaxPooling
from utils.str2bool import str2bool


def main(args):
    df = pd.read_csv(args.path2testdata, delimiter="\t")
    premise, hypothesis = df["premise"].tolist(), df["hypothesis"].tolist()
    targets_placeholder = ["contradiction" for _ in range(len(hypothesis))]
    
    data_prepper = data_prep.DataPrep()

    emb = PretrainedEmbedding(
        "/fp/projects01/ec30/IN5550/magnuspn/40/model.bin"
    )

    collate_fn = CollateFuncPacking(
        pad_idx = emb.get_pad_idx(), max_len=args.max_length
    )

    print("Creating training and validation dataset.")
    train_dataset = NLIDataset(
        premise=premise,
        hypothesis=hypothesis,
        targets=targets_placeholder,
        embedding=emb,
        data_prepper=data_prepper,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4,
        collate_fn = collate_fn
    )

    pooling_func = MaxPooling

    model = Model(
        emb,
        pooling_func,
        args)
    
    # load model
    model.load_state_dict(torch.load(
        "/fp/projects01/ec30/magnuspn/in5550/obligatory2/lstm_best.bin"))
        #"/fp/projects01/ec30/magnuspn/in5550/obligatory2/output/lstm_best.bin"))
    model.to(args.device)
    # put model in eval mode
    model.eval()
    predictions = []
    for premise, hypothesis, _, premise_lens, hypothesis_lens in tqdm(train_dataloader):
        
        y_pred = model(
            premise.to(args.device), hypothesis.to(args.device),
            premise_lens, hypothesis_lens
        )
        predictions += y_pred.argmax(dim=1).tolist()

    df["label"] = predictions
    df.to_csv("predictions.tsv.gz", compression="gzip", sep="\t")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path2testdata", type=str)
    parser.add_argument("--hidden_size", action="store", type=int, default=1024)
    parser.add_argument("--num_layers", action="store", type=int, default=1)
    parser.add_argument("--dropout", action="store", type=float, default=0.15)
    parser.add_argument("--batch_size", action="store", type=int, default=1024)
    parser.add_argument("--max_length", action="store", type=int, default=128)
    parser.add_argument("--freeze", action="store", type=bool, default=True)
    parser.add_argument("--bidirectional", action="store", type=str2bool, default=True)
    parser.add_argument("--n_classes", action="store", type=int, default=3)
    parser.add_argument("--device", action="store", type=str, default="cpu")
    parser.add_argument("--pooling", action="store", type=str, default="max")
    parser.add_argument("--rnn_type", action="store", type=str, default="lstm")
    parser.add_argument("--pooling_multiplier", action="store", type=int, default=1)
    args = parser.parse_args()
    main(args)
