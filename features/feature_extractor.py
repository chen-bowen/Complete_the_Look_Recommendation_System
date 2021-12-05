import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from dataset.create_dataloader import dataloader
from torchvision import models

IMAGE_TYPE = "product"
def extractor():
    """
    Import data loader with the batches. Go through each batch and pass through ResNet. Features are extracted from
    the last layer for each image in each batch. At the very end each batch is stacked and you are left with tensor
    "all_features" of shape (batches,batch_size,features)

    Instead of stack we could just concat all the tensors together which would leave us with a shape of (samples,features)
    """
    data_loader = dataloader()
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()
    transforms = torchvision.transforms.Resize(256)
    all_features = []

    for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
        X = transforms(batch)  # resizes to 256 X 256 for ResNet
        X = X.float()
        with torch.no_grad():
            batch_features = resnet(X)
            all_features.append(batch_features)

    all_features = torch.stack(all_features)
    breakpoint()
    with open(f'{IMAGE_TYPE}_features.pickle', 'wb') as handle:
        pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_features


if __name__ == "__main__":
    extractor()
