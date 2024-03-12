# GaitMix 
The PyTorch implementation of our IJCAI 2024 paper "GaitMix: Integrating Multi-View GEI and 3D Pose for Human Attribute Recognition"

We introduce GaitMix, a human attribute recognition model integrating multi-view GEI and 3D pose. Illustrated in the below Figure, our model has four architectural components: a GEI encoder, a 3D pose encoder, a dual-level fusion module, and an estimation module.

<img src="./img/model.jpg" width="800" height="400"/>

## Updates
[Updated on 03/12/2023] Revised the overall code & read me file.
[Updated on 03/12/2024] Released a code for GaitMix.

## Example Results
Some human attribute recognition results on the OUMVLP dataset. The left value indicates ground age and the right value is estimated age.

<img src="./img/examples.jpg" width="900" height="1600"/>

## Training GaitMix
The training of our GaitMix includes four stages: 1. download the datasets (i.e., OUMVLP and OUMVLP-Mesh); 2. extract three features; 3. split data into train, validation, and test set; 4. Train GaitMix.

1. Data preparing
We use both OUMVLP and OUMVLP-Mesh datasets in experiments and you can download the dataset here: <http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html>

2. Feature extraction
   
We extract three types from two modalities (e.g., GEI and 3D pose)

- GEI features
- HC3D features
- gait features

You can extract them by executing FeatureExtraction.py

```
python FeatureExtraction.py
```

3. Data Split
   
We split the dataset to train, validate, and test sets following the original protocol. You can split dataset by executing DataSplit.py

```
python DataSplit.py
```

4. Train and test the model

If you want to test our model, you need a training weight file, data keys, and extracted features from the mentioned procedure.
You can train and test our model by conducting Train.py

```
python Train.py
```

