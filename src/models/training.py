import numpy as np
import torch
import torch.optim as optim
from src.config import config as cfg
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
from src.losses.loss_function import TripletLoss
from src.models.Model import CompatibilityModel
from src.utils.model_utils import init_weights
from tqdm import tqdm


def train_compatibility_model(num_epochs=10, batch_size=32):
    """train compatibility model with the triplets data"""
    
    model = CompatibilityModel()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = FashionCompleteTheLookDataloader(batch_size=batch_size).data_loader()

    # freeze the base model part of the compatibility model
    for name, param in model.named_parameters():
        if name.split(".")[0] == "base_model":
            param.requires_grad = False

    # initialize weights
    model.apply(init_weights)
    model.train()


    # # compile the model, define loss and optimizer using JIT
    # model = torch.jit.script(model).to(cfg.device)
    model = model.to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # criterion = torch.jit.script(TripletLoss())
    criterion = TripletLoss()

    # training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_epoch = []

        for i, (anchor, positive, negative) in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):
            # send triplets to device
            
            anchor = anchor.to(cfg.device)
            positive = positive.to(cfg.device)
            negative = negative.to(cfg.device)
            breakpoint()
            # forward pass through the model and obtain features for the triplets
            anchor_features = model(anchor)
            positive_features = model(positive)
            negative_features = model(negative)

            # zero out the accumulated gradients
            optimizer.zero_grad()

            # calculate loss and backward pass through the model
            loss = criterion(anchor_features, positive_features, negative_features)
            loss.backward()

            # update the weights
            optimizer.step()

            # append batch loss to epoch loss
            loss_epoch.append(loss.cpu().detach().numpy())

        # print training loss progress
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, num_epochs, np.mean(loss_epoch)))

    # save the trained model to the models directory
    torch.save(
        {"model_state_dict": model.state_dict(), "optimzier_state_dict": optimizer.state_dict()},
        f"{cfg.TRAINED_MODEL_DIR}/trained_compatibility_model.pth",
    )


if __name__ == "__main__":
    train_compatibility_model()
