# hawq-jet-tagging

## Installation
Clone the repository and HAWQ submodule.
```bash
git clone https://github.com/jicampos/hawq-jet-tagging.git
cd hawq-jet-tagging 
git submodule init 
git submodule update 
```

Set up conda environment.
```bash 
conda env create -f environment.yml
conda activate hawq-env
```

Download [LHC Jet dataset](https://paperswithcode.com/dataset/hls4ml-lhc-jet-dataset).
```bash
mkdir dataset

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P dataset
wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P dataset

tar -zxf dataset/hls4ml_LHCjet_100p_train.tar.gz -C dataset
tar -zxf dataset/hls4ml_LHCjet_100p_val.tar.gz -C dataset
```

## Training and Configuration
Predefined bit configurations are provided under `config` directory. For custom bit configurations, specify the layer name followed by the number of bits inside config files:
```yaml
model:
    quant_input: # layer name
        activation_bit: 6
    dense_1:     # layer name
        weight_bit: 8
        bias_bit: 8
```

Specify the bit configuration file using `--config` for training (Default: `config/config_6bit.yml`)
```bash
python train_jettagger.py --config config/config_6bit.yml \
    --epochs 25 \
    --lr 0.01 \ 
    --data dataset \
    --batch-size 1024 \ 
    --save-path checkpoints
```
To train with Batch Normalization or L1 Regularization, use `--batch-norm` and `--l1`. For all training and logging options, use:
```bash
python train_jettagger.py --help
```
