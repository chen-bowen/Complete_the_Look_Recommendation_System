import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from src.config import config as cfg
from src.dataset.Dataloader import FashionProductSTLDataloader
from torchvision import models


class StyleEmbedding:
    """
    Feature extractor that generates different features according to data_loader and task
    """

    def __init__(self, data_loader, dataset_name):
        self.data_loader = data_loader
        self.dataset_name = dataset_name

    def similar_product_embedding(self):
        """
        Import data loader with the batches. Go through each batch and pass through ResNet. Features are extracted from
        the last layer for each image in each batch. At the very end each batch is stacked and you are left with tensor
        "all_features" of shape (batches,batch_size,features)
        """
        data_loader = self.data_loader

        # get pretrain model and remove the classification layer
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        resnet.eval()
        transforms = torchvision.transforms.Resize(256)
        all_features = []

        for batch in tqdm.tqdm(data_loader):
            X = transforms(batch)  # resizes to 256 X 256 for ResNet
            X = X.float()
            with torch.no_grad():
                batch_features = resnet(X)
                all_features.append(batch_features)

        all_features = torch.cat(all_features)

        # save all features to a pickle file
        with open(
            f"{cfg.PACKAGE_ROOT}/features/{self.dataset_name}_features.pickle", "wb"
        ) as handle:
            pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_features


if __name__ == "__main__":
    StyleEmbedding(
        data_loader=FashionProductSTLDataloader().data_loader(), dataset_name="product"
    ).similar_product_embedding()
