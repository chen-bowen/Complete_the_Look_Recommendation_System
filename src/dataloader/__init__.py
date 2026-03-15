"""Dataloaders for STL, CTL, Street2Shop, and Polyvore."""

from src.dataloader.data_loaders import (FashionCompleteTheLookDataloader,
                                         FashionProductSTLDataloader,
                                         Street2ShopDataloader)

__all__ = [
    "FashionCompleteTheLookDataloader",
    "FashionProductSTLDataloader",
    "Street2ShopDataloader",
]
