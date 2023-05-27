import data.data_prep as data_prep
from data.data_utils import read_tsv_lists
from data.datasplit import stratified_split
from utils.collate_fn import CollateFuncPacking
from data.nli_dataset import NLIDataset
import torch
from typing import Tuple


def data_pipeline(args, emb) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepares data and constructs dataloaders based on args.

    Args:
        args : Arguments from ArgsParse. 
        emb : Pretrained embedding.

    Raises:
        Exception: If args.prep is not raw, pos or lemmatized.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: train and val dataloader.
    """
    if args.prep not in ["raw", "pos", "lemmatized"]:
        raise Exception("Prep must be either raw, pos or lemmatized")

    # select correct dataprep based on args
    if args.prep == "raw":
        data_prepper = data_prep.DataPrep()
    if args.prep == "pos": 
        data_prepper = data_prep.DataPrepPos()
    if args.prep == "lemmatized":
        data_prepper = data_prep.DataPrepLemma()

    print("Loading data.")
    # read and split data
    if args.prep in ["raw", "lemmatized"]:
        targets, premise, hypothesis, _, _ = read_tsv_lists(args.path2data)
    else:
        targets, _, _, premise, hypothesis = read_tsv_lists(args.path2data)

    if args.samples2keep:
        targets, premise, hypothesis = \
            targets[:args.samples2keep], premise[:args.samples2keep], hypothesis[:args.samples2keep]
        
    targets_train, targets_val, premise_train, premise_val, hypo_train, hypo_val = \
        stratified_split(targets, premise, hypothesis, test_size=args.test_size)

    print("Data loaded.")
    
    # init collate func for padding uneven sequence lengths 
    collate_fn = CollateFuncPacking(
        pad_idx = emb.get_pad_idx(), max_len=args.max_length
    )

    print("Creating training and validation dataset.")
    train_dataset = NLIDataset(
        premise=premise_train,
        hypothesis=hypo_train,
        targets=targets_train,
        embedding=emb,
        data_prepper=data_prepper,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn = collate_fn
    )

    val_dataset = NLIDataset(
        premise=premise_val,
        hypothesis=hypo_val,
        targets=targets_val,
        embedding=emb,
        data_prepper=data_prepper,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn = collate_fn
    )

    print("Training and validation dataset constructed.")

    return train_dataloader, val_dataloader
