from argparse import ArgumentParser
import time
from train_fn import train
import torch
from models.model import Model
from models.pooling import get_pooling
from utils.seed_everything import seed_everything
from data.data_pipeline import data_pipeline
from pretrained_embedding.pretrained_embedding import get_embedding
from utils.str2bool import str2bool


def main(args):
    # seed everything to make training more deterministic
    print(f"Seeding everything with seed = {args.seed}")
    seed_everything(args.seed)

    print("Loading embedding.")
    # load and init pretrained embedding
    emb = get_embedding(args)
    print("Embedding loaded.")
    # create dataloaders
    train_dataloader, val_dataloader = data_pipeline(args, emb)

    print("Training and validation dataset constructed.")
    
    print(f"Pooling function set to {args.pooling}.")
    pooling_func = get_pooling(args)
    # init model
    model = Model(
        emb,
        pooling_func,
        args,
    )
    # AdamW as optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05
    )
    # use ExponetialLR as learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9)
    # start training and measure training duration
    start_time = time.time()
    train(model, optimizer, train_dataloader, scheduler, 
          val_dataloader = val_dataloader, n_epochs=args.epochs,
          device = args.device)
    print(f" Elapsed training time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path2data", action="store", type=str, default="/fp/projects01/ec30/IN5550/obligatories/2/mnli_train.tsv.gz")
    parser.add_argument("--prep", action="store", type=str, default="raw")
    parser.add_argument("--path2embedding", action="store", type=str, default="/fp/projects01/ec30/magvic_large_files/modelbins/40/model.bin")
    parser.add_argument("--fasttext", action="store", type=bool, default=False)
    parser.add_argument("--hidden_size", action="store", type=int, default=256)
    parser.add_argument("--num_layers", action="store", type=int, default=1)
    parser.add_argument("--dropout", action="store", type=float, default=0.15)
    parser.add_argument("--batch_size", action="store", type=int, default=512)
    parser.add_argument("--lr", action="store", type=float, default=0.03)
    parser.add_argument("--epochs", action="store", type=int, default=3)
    parser.add_argument("--max_length", action="store", type=int, default=128)
    parser.add_argument("--test_size", action="store", type=int, default=10_000)
    parser.add_argument("--seed", action="store", type=int, default=9001)
    parser.add_argument("--samples2keep", action="store", type=int, default=None)
    parser.add_argument("--freeze", action="store", type=str2bool, default=True)
    parser.add_argument("--n_classes", action="store", type=int, default=3)
    parser.add_argument("--device", action="store", type=str, default="cpu")
    parser.add_argument("--pooling", action="store", type=str, default="mean")
    parser.add_argument("--concat", action="store", type=bool, default=False)
    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k} = {v}")

    main(args)
