from torch.nn import functional as F
from evaluation.eval_fns import accuracy_packed_seqs
from tqdm import tqdm
from utils.eary_stopping import EarlyStopping
import logging


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def train(
    model, optimizer, train_dataloader, scheduler, 
    val_dataloader, n_epochs = 5, 
    loss_fn = F.cross_entropy, device = "cpu"
):
    """
    Training loop for ANNs and RNNs. Implements early stopping and calculates
    accuracy for the training and validation set at the end of each epoch. 

    Args:
        model : model to train. 
        optimizer : optimizer to use. 
        train_dataloader : dataloader for training set. 
        scheduler : learning rate scheduler. 
        val_dataloader : dataloader for validation set. 
        n_epochs : Numbers of epoch to train for. Defaults to 5.
        loss_fn : Loss function. Defaults to F.cross_entropy.
        device : What device to use, cuda or cpu. Defaults to "cpu".

    Returns:
        Trained model 
    """
    accum_loss = []
    train_acc = []
    val_acc = []
    model.to(device)
    early_stopper = EarlyStopping()
    prev_acc = 0.0
    
    logger.info(f"Starting training with {n_epochs} epochs.")
    for epoch in range(n_epochs):
        # put model in train mode, if drop out is included in forward this will be activated
        model.train()

        # get data and targets from the dataloader, these are put to the correct device
        for premise, hypothesis, targets, premise_lens, hypothesis_lens in tqdm(train_dataloader):
            # zero out gradients
            optimizer.zero_grad()
            # predict based on premise and hypothesis 
            y_pred = model(
                premise.to(device), hypothesis.to(device),
                premise_lens, hypothesis_lens
            )

            # calcualte loss
            loss = loss_fn(
                y_pred, 
                targets.to(device)
            ).mean()
            # keep track of loss
            accum_loss.append(loss.item())
            # calculate grads 
            loss.backward()
            # update weights w.r.t. grads 
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25, error_if_nonfinite=True)
            optimizer.step()
        # calculate accuracy on training data (yes, this is kinda slow..)
        epoch_train_acc = accuracy_packed_seqs(
                model, train_dataloader, device
        )
        # train accuracy is appended to list
        train_acc.append(epoch_train_acc)
        logger.info(f"Epoch {epoch} - train loss: {accum_loss[-1]:.3f} - train acc: {train_acc[-1]:.3f}")
        # calculate accuracy on validation data
        epoch_val_acc = accuracy_packed_seqs(
            model, val_dataloader, device
        )
        val_acc.append(epoch_val_acc)
        logger.info(f"Val acc: {val_acc[-1]:.3f}")
        # if current model is better then previous best; save state dict
        if epoch_val_acc > prev_acc:
                best_state = model.state_dict()
                prev_acc = epoch_val_acc
        # early stopping returns false if model has not improved for k epochs
        if early_stopper.early_stop(epoch_val_acc):
            break

        # adjust learning rate
        scheduler.step()
    # after training; set model to best version in train loop
    model.load_state_dict(best_state)

    return model
