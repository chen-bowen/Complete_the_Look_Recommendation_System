import pickle
from FashionDataset import FashionProductDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import config as cfg

BATCH_SIZE = cfg.batch_size
FOLDER = "product"
def create_dataloader():
    file = open(f'{FOLDER}_df.pickle', 'rb')
    X_train = pickle.load(file)
    file.close()

    # file = open('FILENAME_Y_train', 'rb')
    # Y_train = pickle.load(file)
    # file.close()

    print("Creating data loader...")

    product_dataset = FashionProductDataset(X_train, transform=transforms.Compose([transforms.ToTensor()]))
    data_loader = DataLoader(product_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Data loader complete. Ready for use.")

    return data_loader