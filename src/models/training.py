import numpy as np
import torch
import torch.optim as optim
from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
from src.losses.loss_function import TripletLoss
from src.models.Model import CompatibilityModel
from src.utils.model_utils import init_weights
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train_compatibility_model(num_epochs=2, batch_size=32):
    """train compatibility model with the triplets data"""

    model = CompatibilityModel()
    train_dataloader = FashionCompleteTheLookDataloader(batch_size=batch_size).data_loader()

    # freeze the base model part of the compatibility model
    for name, param in model.named_parameters():
        if name.split(".")[0] == "base_model":
            param.requires_grad = False

    # initialize weights
    model.apply(init_weights)
    model.train()

    # compile the model, define loss and optimizer using JIT
    model = model.to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = TripletLoss()
    print("You are using device: " + str(cfg.device))

    # training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        loss_epoch = []
        print(cfg.device)
        for i, (anchor, positive, negative) in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            # send triplets to device

            # send triplets to device
            anchor = anchor.to(cfg.device)
            positive = positive.to(cfg.device)
            negative = negative.to(cfg.device)
            # forward pass through the model and obtain features for the triplets

            anchor_features = model(anchor)
            positive_features = model(positive)
            negative_features = model(negative)

            # calculate loss and backward pass through the model
            loss = criterion(anchor_features, positive_features, negative_features)
            loss.backward(inputs=tuple(model.embedding_layers.parameters()), retain_graph=True)

            # update the weights
            optimizer.step()

            # append batch loss to epoch loss
            loss_epoch += loss.cpu().detach().numpy()

        # print training loss progress
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, num_epochs, np.mean(loss_epoch)))

        # save the trained model to the models directory
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(),
            },
            f"{cfg.TRAINED_MODEL_DIR}/trained_compatibility_model_epoch{epoch}.pth",
        )


if __name__ == "__main__":
    train_compatibility_model(batch_size=cfg.BATCH_SIZE)
