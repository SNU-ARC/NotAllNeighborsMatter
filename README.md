# NotAllNeighborsMatter

## Environments

```
- gcc-7
- cuda 11.2
```

## Setup

```
conda create -n NotAllNeighborsMatter python=3.8
conda activate NotAllNeighborsMatter
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f <https://download.pytorch.org/whl/torch_stable.html>
pip install ninja
cd s3dis_scannet
pip install -r requirements.txt
pip install protobuf==3.20.0
pip install scikit-learn==1.0.2
cd ../libraries/MinkowskiEngine
python setup.py install
cd ../libraries/torchsparse
python setup.py install
cd ../libraries/cumm
pip install --editable .
pip install ccimport==0.3.7
python setup.py develop
cd ../libraries/spconv
python setup.py develop
python (Open python prompt)
import spconv (This will build cumm and spconv)

```

## S3DIS

### Prepare Dataset (Brought from [here](https://github.com/chrischoy/SpatioTemporalSegmentation/tree/master#stanford-3d-dataset))

1. Download the stanford 3d dataset (Stanford3dDataset_v1.2.zip) from [the website](http://buildingparser.stanford.edu/dataset.html) to `./s3dis_scannet/dataset/s3dis`
2. Preprocess
    - Modify the input and output directory accordingly in `lib/datasets/preprocessing/stanford.py`
    - And run `python -m lib.datasets.preprocessing.stanford`
3. Train
`./scripts/train_stanford.sh 0 "-default" "--stanford3d_path ./dataset/s3dis/preprocessing"`

### Train

To train baseline model from scratch, run below command.

We also provide checkpoints of baseline models in `./checkpoints`.

```
LOG_DIR=PATH_TO_OUTPUT_DIR ./scripts/train_stanford.sh $GPU_ID --backend=[spconv, mink, torchsparse]
```

- If you would like to resume training from the latest checkpoint, add `-resume=PATH_TO_OUTPUT_DIR` following the train command

### Retrain

For given prune_edge configuration, run below command to retrain baseline checkpoint to recover the accuracy loss.

```
./scripts/retrain_stanford.sh $GPU_ID --backend=[spconv, mink, torchsparse]  --weights=PATH_TO_BASELINE_CKPT --prune_edge "{'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}"
```

- e.g., To retrain pruned baseline model with `--prune_edge "{'0':4,'1':3,'2':2,'3':2,'4':1,'5':0,'6':0,'7':0}"` with spconv backend, run
    
    ```
    ./scripts/retrain_scannet.sh 0 --backend=spconv --weights=PATH_TO_BASELINE_CKPT --prune_edge "{'0':4,'1':3,'2':2,'3':2,'4':1,'5':0,'6':0,'7':0}"
    ```
    

### Test

```
./scripts/test_stanford.sh $GPU_ID --backend=[spconv, mink, torchsparse] --weights=PATH_TO_CKPT --prune_edge "{'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}"
```

- To run model with half precision, add `-fp16` following the test command

## ScannetV2

### Prepare Dataset (Brought from [here](https://github.com/chrischoy/SpatioTemporalSegmentation/tree/master#scannet-training))

1. Download the ScanNet dataset from [the official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation) to `./dataset/scannet`. You need to sign the terms of use.
2. Next, preprocess all scannet raw point cloud with the following command after you set the path correctly.
`python -m lib.datasets.preprocessing.scannet`
3. Train the network with
`export BATCH_SIZE=N; ./scripts/train_scannet.sh 0 -default "--scannet_path ./dataset/scannet/preprocessing"`
    - Modify the `BATCH_SIZE` accordingly.
    - The first argument is the GPU id and the second argument is the path postfix and the last argument is the miscellaneous arguments.

### Train

To train baseline model from scratch, run below command.

We also provide checkpoints of baseline models in `./checkpoints`.

```
./scripts/train_scannet.sh $GPU_ID --backend=[spconv, mink, torchsparse] PATH_TO_OUTPUT_DIR
```

- If you would like to resume training from the latest checkpoint, add `-resume PATH_TO_OUTPUT_DIR` following the train command

### Retrain

For given prune_edge configuration, run below command to retrain baseline checkpoint to recover the accuracy loss.

```
./scripts/retrain_scannet.sh $GPU_ID --backend=[spconv, mink, torchsparse] --weights=PATH_TO_BASELINE_CKPT --prune_edge "{'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}"
```

- e.g., To retrain pruned baseline model with `--prune_edge "{'0':4,'1':3,'2':2,'3':2,'4':1,'5':0,'6':0,'7':0}"` with spconv backend, run
    
    ```
    ./scripts/retrain_scannet.sh 0 --backend=spconv --weights=PATH_TO_BASELINE_CKPT --prune_edge "{'0':4,'1':3,'2':2,'3':2,'4':1,'5':0,'6':0,'7':0}"
    ```
    

### Test

```
./scripts/test_scannet.sh $GPU_ID --backend=[spconv, mink, torchsparse] --weights=PATH_TO_CKPT --prune_edge "{'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}"
```

- To run model with half precision, add `-fp16` following the test command


## Checkpoints
Checkpoints are uploaded in `./checkpoints`.
| Dataset     | Library          | Baseline      | Lossless     | Lossy     | 
| ----------- | ----------       | ------------- | ------------ | --------- |
|  S3DIS      | Spconv           | 63.107        | 63.464       | 62.052    |
|  S3DIS      | MinkowskiEngine  | 63.547        | 63.471       | 62.801    |
| ScannetV2   | Spconv           | 72.354        | 72.413       | 71.461    |
| ScannetV2   | MinkowskiEngine  | 72.341        | 72.108       | 71.340    |
