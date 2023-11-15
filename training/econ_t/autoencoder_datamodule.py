import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils_pt import normalize
from torch.utils.data import TensorDataset


ARRANGE = torch.tensor(
    [
        28,
        29,
        30,
        31,
        0,
        4,
        8,
        12,
        24,
        25,
        26,
        27,
        1,
        5,
        9,
        13,
        20,
        21,
        22,
        23,
        2,
        6,
        10,
        14,
        16,
        17,
        18,
        19,
        3,
        7,
        11,
        15,
        47,
        43,
        39,
        35,
        35,
        34,
        33,
        32,
        46,
        42,
        38,
        34,
        39,
        38,
        37,
        36,
        45,
        41,
        37,
        33,
        43,
        42,
        41,
        40,
        44,
        40,
        36,
        32,
        47,
        46,
        45,
        44,
    ]
)

ARRANGE_MASK = torch.tensor(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
)


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, data_file, data_dir=None, batch_size=500, num_workers=8) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.train_bs = batch_size
        self.test_bs = batch_size
        self.num_workers = num_workers
        self.calq_cols = [f"CALQ_{i}" for i in range(48)]
        self.valid_split = 0.2
        self.val_max = None
        self.val_sum = None
        self.max_data = None
        self.sum_dat = None
        self.train_data = None
        self.val_data = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")
        parser.add_argument("--data_dir", 
                            type=str, 
                            default=os.path.join(
                                os.environ["HAWQ_JET_TAGGING"],
                                "data/econ/raw"
                            )
        )
        parser.add_argument("--data_file", 
                            type=str, 
                            default= os.path.join(
                                os.environ["HAWQ_JET_TAGGING"], 
                                "data/econ/preprocess/hgcal22data_signal_driven_ttbar_v11_nELinks5.npy"
                            )
        )
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=500)
        return parent_parser

    def mask_data(self, data):
        """
        Mask rows where occupancy is zero
        """
        return data[data[self.calq_cols].astype("float32").sum(axis=1) != 0]

    def load_data_dir(self):
        """
        Read and concat all csv files in the data directory into a single
        dataframe
        """
        print(f"Reading files {os.listdir(self.data_dir)}")
        files = os.listdir(self.data_dir)
        data = pd.concat(
            [
                pd.read_csv(os.path.join(self.data_dir, file))
                for file in [files[0]]
            ]
        )
        data = self.mask_data(data)
        data = data[self.calq_cols].astype("float32")
        print(f"Input data shape: {data.shape}")

        return data

    def prep_input(self, norm_data, shape=(1, 8, 8)):
        """
        Prepare the input data for the model
        """
        input_data = norm_data[:, ARRANGE]
        input_data[:, ARRANGE_MASK == 0] = 0  # zero out repeated entries
        shaped_data = input_data.reshape(len(input_data), shape[0], shape[1], shape[2])
        print(f"Prepped shaped data shape: {shaped_data.shape}")
        return shaped_data

    def get_val_max_and_sum(self):
        shaped_data, max_data, sum_data = self.process_data(save=False)
        max_data = max_data / 35.0  # normalize to units of transverse MIPs
        sum_data = sum_data / 35.0  # normalize to units of transverse MIPs
        val_index = np.arange(int(len(shaped_data) * self.valid_split))
        self.val_max = max_data[val_index]
        self.val_sum = sum_data[val_index]
        return self.val_max, self.val_sum

    def process_data(self, save=True):
        """
        Only need to run once to prepare the data and pickle it
        """
        data = self.load_data_dir()
        norm_data, max_data, sum_data = normalize(data.values.copy())
        shaped_data = self.prep_input(norm_data)
        if save:
            np.save(self.data_file, shaped_data)
        return shaped_data, max_data, sum_data

    # PyTorch Lightning specific methods
    def setup(self, stage):
        """
        Load data from provided npy data_file
        """
        shaped_data = np.load(self.data_file)
        print(f"Loaded shaped data shape: {shaped_data.shape}")
        print(f"Loaded shaped data datatype: {shaped_data.dtype}")

        self.train_data = shaped_data[int(len(shaped_data) * self.valid_split) :]
        self.val_data = shaped_data[: int(len(shaped_data) * self.valid_split)]

    def train_dataloader(self):
        """
        Return the training dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Return the validation dataloader
        """
        # Take the first valid_split% of the data as validation data
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.test_bs, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """
        Return the test dataloader
        """
        return self.val_dataloader()

    def dataloaders(self):
        """
        Return train and test as Tensor dataloaders. Used for metrics, not training
        """
        val_data_tensor = torch.Tensor(self.val_data)
        val_dataset = TensorDataset(val_data_tensor, val_data_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.test_bs, shuffle=False, num_workers=self.num_workers
        )
    
        train_data_tensor = torch.Tensor(self.train_data)
        train_dataset = TensorDataset(train_data_tensor, train_data_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers
        )
        return train_loader, val_loader
