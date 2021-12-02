import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import make_data
from make_data import create_dataloader


def extractor():
    """
    Import data loader with the batches. Go through each batch and pass through ResNet. Features are extracted from 
    the last layer for each image in each batch. At the very end each batch is stacked and you are left with tensor
    "all_features" of shape (batches,batch_size,features)

    Instead of stack we could just concat all the tensors together which would leave us with a shape of (samples,features)
    """
    data_loader = make_data.create_dataloader()
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()
    # target_layer = resnet._modules.get('avgpool')
    transforms = torchvision.transforms.Resize(256)
    all_features = []
    
    for (batch_idx, batch) in enumerate(data_loader):
        X = transforms(batch) # resizes to 256 X 256 for ResNet
        X = X.float()
        with torch.no_grad():
            batch_features = resnet(X)
            # batch_features = batch_features.unsqueeze(0)
            all_features.append(batch_features)
        # print(round(batch_idx/len(data_loader)*100,2),"percent complete",end = '\r',flush=True)
        print(batch_idx+1, "batch(s) extracted")
    all_features = torch.stack(all_features)
    # breakpoint()
    return all_features
    
if __name__ == "__main__":
    extractor()