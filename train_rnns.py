from argparse import ArgumentParser
import time
from train_fn import train
import torch
from models.rnn import Model
from models.lstm_pooling import get_pooling
from utils.seed_everything import seed_everything
from data.data_pipeline import data_pipeline
from pretrained_embedding.pretrained_embedding import get_embedding
from utils.str2bool import str2bool
import logging


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(args):
    print("\n")
    logger.info("#"*60)
    logger.info(f"RNN type is set to {args.rnn_type}")
    logger.info(f"Seeding everything with seed = {args.seed}")
    seed_everything(args.seed)
    
    # load and init pretrained embedding
    emb = get_embedding(args)
    
    train_dataloader, val_dataloader = data_pipeline(args, emb)

    logger.info(f"Pooling function set to {args.pooling}.")
    pooling_func = get_pooling(args)

    model = Model(
        emb,
        pooling_func,
        args,
    )

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
    # start training and measure duration
    start_time = time.time()
    trained_model = train(model, optimizer, train_dataloader, scheduler, 
          val_dataloader = val_dataloader, n_epochs=args.epochs,
          device = args.device)
    logger.info(f" Elapsed training time: {time.time() - start_time}")
    # save model if args.save
    if args.save:
        torch.save(trained_model.state_dict(), f"/fp/homes01/u01/ec-magnuspn/{args.model_name}.bin")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--save", action="store", type=str2bool, default=True)
    parser.add_argument("--path2data", action="store", type=str, default="/fp/projects01/ec30/IN5550/obligatories/2/mnli_train.tsv.gz")
    parser.add_argument("--prep", action="store", type=str, default="raw")
    parser.add_argument("--path2embedding", action="store", type=str, default="/fp/projects01/ec30/magvic_large_files/modelbins/40/model.bin")
    parser.add_argument("--hidden_size", action="store", type=int, default=1024)
    parser.add_argument("--num_layers", action="store", type=int, default=1)
    parser.add_argument("--dropout", action="store", type=float, default=0.15)
    parser.add_argument("--batch_size", action="store", type=int, default=1024)
    parser.add_argument("--lr", action="store", type=float, default=0.03)
    parser.add_argument("--epochs", action="store", type=int, default=25)
    parser.add_argument("--max_length", action="store", type=int, default=128)
    parser.add_argument("--test_size", action="store", type=int, default=10_000)
    parser.add_argument("--seed", action="store", type=int, default=9001)
    parser.add_argument("--samples2keep", action="store", type=int, default=None)
    parser.add_argument("--freeze", action="store", type=bool, default=True)
    parser.add_argument("--bidirectional", action="store", type=str2bool, default=True)
    parser.add_argument("--n_classes", action="store", type=int, default=3)
    parser.add_argument("--device", action="store", type=str, default="cuda")
    parser.add_argument("--pooling", action="store", type=str, default="combined")
    parser.add_argument("--rnn_type", action="store", type=str, default="lstm")
    parser.add_argument("--fasttext", action="store", type=bool, default=False)
    parser.add_argument("--pooling_multiplier", action="store", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
