import os
from typing import Optional
from glob import glob
import pytorch_lightning as pl
from src.peer_x.data.dataset import TestDataset, TrainDataset, ValDataset
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None:
        self.input_img_paths = glob(os.path.join(self.data_dir, "*.png"), recursive=True)


    def setup(self) -> None:
        self.train = TrainDataset()
        self.val = ValDataset()
        self.test = TestDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        pass
