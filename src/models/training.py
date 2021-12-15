import numpy as np
import torch
import torch.optim as optim
from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
from src.models.Model import CompatibilityModel
from src.utils.image_utils import plot_learning_curves
from src.utils.model_utils import init_weights
from tqdm import tqdm

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train_compatibility_model(num_epochs=1, batch_size=32):
    """train compatibility model with the triplets data"""
    model = CompatibilityModel()

    train_dataloader = FashionCompleteTheLookDataloader(batch_size=batch_size).triplet_data_loader()
    validation_dataloader = FashionCompleteTheLookDataloader(
        batch_size=max(batch_size // 9, 1), image_type="validation"
    ).triplet_data_loader()

    # freeze the base model part of the compatibility model
    for name, param in model.named_parameters():
        if name.split(".")[0] == "base_model":
            param.requires_grad = False

    # initialize weights
    model.apply(init_weights)
    model.train()

    # compile the model, define loss and optimizer using JIT

    model = torch.jit.script(model).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.jit.script(torch.nn.TripletMarginLoss(margin=0.2)).to(cfg.device)
    training_losses = []
    validation_losses = []

    # training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        for i, (anchor, positive, negative) in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            # set gradient accumulation to 0
            optimizer.zero_grad(set_to_none=True)

            # get triplet loss and back update
            loss = get_triplet_loss(anchor, positive, negative, criterion, model)
            loss.backward()

            # update the weights
            optimizer.step()

            # append batch loss to epoch loss
            if i % 100 == 0:
                training_losses.append(loss.cpu().detach().numpy())
                # get validation loss
                model.eval()
                with torch.no_grad():
                    try:
                        iterator = iter(validation_dataloader)
                        anchor_val, positive_val, negative_val = iterator.next()
                    except:  # if reaches the end, reset
                        iterator = iter(validation_dataloader)
                        anchor_val, positive_val, negative_val = iterator.next()

                    # calculate validation loss and backward pass through the model
                    loss_valid = get_triplet_loss(
                        anchor_val, positive_val, negative_val, criterion, model
                    )
                    validation_losses.append(loss_valid.cpu().detach().numpy())

                print(
                    "\nAvg Training Loss: {:.4f}, Step Training Loss: {:.4f}, Avg Validation Loss: {:.4f}, Step Validation Loss: {:.4f}\n".format(
                        np.mean(training_losses),
                        training_losses[-1],
                        np.mean(validation_losses),
                        validation_losses[-1],
                    )
                )

        # save the trained model to the models directory
        torch.save(
            {"model_state_dict": model.state_dict()},
            f"{cfg.TRAINED_MODEL_DIR}/trained_compatibility_model_epoch{epoch}.pth",
        )
        plot_learning_curves(training_losses, validation_losses)


def get_triplet_loss(anchor, positive, negative, criterion, model):
    """Given anchor, positive, negative, obtain the triplet loss"""
    # send triplets to cfg.device
    anchor = anchor.to(cfg.device)
    positive = positive.to(cfg.device)
    negative = negative.to(cfg.device)

    # forward pass through the model and obtain features for the triplets
    anchor_features = model(anchor)
    positive_features = model(positive)
    negative_features = model(negative)

    # calculate loss and backward pass through the model
    loss = criterion(anchor_features, positive_features, negative_features)

    return loss


if __name__ == "__main__":
    train_compatibility_model(num_epochs=cfg.NUM_EPOCHS, batch_size=cfg.BATCH_SIZE)
    # plot learning curve
