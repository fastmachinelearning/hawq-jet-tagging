import os
import h5py
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

from sklearn import preprocessing


class JetTaggingDataset(Dataset):
    def __init__(self, path, features, preprocess):
        """
        Args:
            path (str): Path to dataset.
            features (list[str]): Load selected features from dataset.
            preprocess (str): Standardize or normalize data.

        Raises:
            RuntimeError: If path is not a directory.
        """
        self.path = path
        self.features = features

        if os.path.isdir(path):
            self.data, self.labels = self.load_data()
        else:
            raise RuntimeError(f"Path is not a directory: {path}")

        if preprocess == "standardize":
            scaler = preprocessing.StandardScaler().fit(self.data)
            self.data = scaler.transform(self.data)
        elif preprocess == "normalize":
            normalizer = preprocessing.Normalizer().fit(self.data)
            self.data = normalizer.transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data(self, files):
        data = np.empty([1, 54])
        labels = np.empty([1, 5])
        files_parsed = 0
        progress_bar = tqdm(files)

        for file in progress_bar:
            file = os.path.join(self.path, file)
            try:
                h5_file = h5py.File(file, "r")

                if files_parsed == 0:
                    feature_names = np.array(h5_file["jetFeatureNames"])
                    feature_names = np.array(
                        [ft.decode("utf-8") for ft in feature_names]
                    )
                    feature_indices = [
                        int(np.where(feature_names == feature)[0])
                        for feature in self.features
                    ]

                h5_dataset = h5_file["jets"]
                # convert to ndarray and concatenate with dataset
                h5_dataset = np.array(h5_dataset, dtype=np.float)
                # separate data from labels
                np_data = h5_dataset[:, :54]
                np_labels = h5_dataset[:, -6:-1]
                # update data and labels
                data = np.concatenate((data, np_data), axis=0, dtype=np.float)
                labels = np.concatenate((labels, np_labels), axis=0, dtype=np.float)
                h5_file.close()
                # update progress bar
                files_parsed += 1
                progress_bar.set_postfix({"files loaded": files_parsed})
            except:
                print(f"Could not load file: {file}")

        data = data[:, feature_indices]
        return data[1:].astype(np.float32), labels[1:].astype(np.float32)

    def load_data(self):
        files = os.listdir(self.path)
        files = [file for file in files if file.endswith(".h5")]
        if len(files) == 0:
            print("Directory does not contain any .h5 files")
            return None
        return self._load_data(files)
