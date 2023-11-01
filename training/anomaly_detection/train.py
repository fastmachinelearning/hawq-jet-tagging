"""
 This script is based on the MLPerf Tiny v0.7 Submission.
 Segments have been added and removed for PyTorch support. 
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import time
import datetime
########################################################################

# get date and time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
second = dt.second

experiment_name = f"{month}.{day}.{year}-{hour}.{minute}.{second}"

########################################################################
# import additional python-library
########################################################################
import numpy 
# from import
from tqdm import tqdm
# original lib
import common as com
from model import AD08
import torch 
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torchinfo
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load("ad08.yaml")
param = param["train"]
########################################################################


########################################################################
# visualizer
########################################################################
# todo: move to utilities 
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         downsample=False,
                         input_dims=640):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames
    mels = 32
    print(f'length of files_list is {len(file_list)}')

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power,
                                                downsample=downsample,
                                                input_dim=input_dims)
        if idx == 0:
            if downsample:
                dataset = numpy.zeros((vector_array.shape[0] * len(file_list), input_dims), float)
            else:
                dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    print("Shape of dataset: {}".format(dataset.shape))
    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files


def dataset_split(dataset, val_split=0.1):
    length = int(len(dataset) * val_split)
    val_dataset = dataset[-length:]
    train_dataset = dataset[:-length]
    return train_dataset, val_dataset
    
########################################################################


########################################################################
# main train.py
########################################################################
if __name__ == "__main__":
    start_time = time.time()

    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        print('Mode is None. Exiting.')
        sys.exit(-1)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    print('Loading base directory list')
    dirs = com.select_dirs(param=param)
    print(dirs)

    print('Looping base directory')
    print(dirs)
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        checkpoint_dir = os.path.join(os.environ["HAWQ_JET_TAGGING"], param["model_directory"], machine_type)
        history_img = os.path.join(checkpoint_dir, experiment_name, f"history_{machine_type}.png")

        if os.path.exists(checkpoint_dir) == False:
            com.logger.info(f"{machine_type} checkpoint directory does not exist. Creating one.")
            os.mkdir(checkpoint_dir)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(files,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            downsample=param["feature"]["downsample"],
                                            input_dims = param["model"]["input_dim"])
        
        train_tensor = torch.Tensor(train_data)
        train_dataset, val_dataset = dataset_split(train_tensor, val_split=param["fit"]["validation_split"])

        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset   = TensorDataset(val_dataset, val_dataset)

        train_dataloader = DataLoader(  # create your dataloader
            dataset=train_dataset, 
            batch_size=param["fit"]["batch_size"], 
            shuffle=param["fit"]["shuffle"]
        )
        val_dataloader = DataLoader(  # create your dataloader
            dataset=val_dataset, 
            batch_size=param["fit"]["batch_size"], 
            shuffle=False
        )

        # train model
        print("============== MODEL TRAINING ==============")
        model = AD08(
            input_shape=param['model']['input_dim'],
            precision=[],
            lr=1e-3,
        )
        torchinfo.summary(model, (2, 64))

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=checkpoint_dir, name=experiment_name)

        # Stop training when model converges
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")

        # Save top-1 checkpoints based on Val/Loss
        top1_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            mode="min",
            dirpath=os.path.join(checkpoint_dir, experiment_name),
            filename=f'model_best',
            auto_insert_metric_name=False,
        )

        trainer = pl.Trainer(
            max_epochs=param["fit"]["epochs"],
            accelerator='auto',
            logger=tb_logger,
            callbacks=[top1_checkpoint_callback, early_stop_callback],
        )

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Load best checkpoint 
        checkpoint_file = os.path.join(checkpoint_dir, experiment_name, 'model_best.ckpt')
        print('Loading checkpoint:', checkpoint_file)
        model.load_state_dict(torch.load(checkpoint_file)['state_dict'])

        trainer.test(model=model, dataloaders=val_dataloader)[0]

        # ------------------------
        # RECORD MODEL METRICS 
        # ------------------------
        elapsed_time = time.time() - start_time
        
        history = {}
        history["loss"] = numpy.array(model.train_step_loss)
        history["val_loss"] = numpy.array(model.validation_step_loss)

        with open(os.path.join(checkpoint_dir, "summary.txt"), "a") as f:
            f.write("=====================================================\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"\tTrain Loss: {history['loss'].mean()}\n")
            f.write(f"\tVal Loss: {history['val_loss'].mean()}\n")
            f.write(f"\tExecution Time (s): {elapsed_time}\n")
            f.close()
        
        visualizer.loss_plot(history["loss"], history["val_loss"])
        visualizer.save_figure(history_img)
        com.logger.info("save_model -> {}".format(os.path.join(checkpoint_dir, experiment_name)))
        print("============== END TRAINING ==============")
