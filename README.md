# GaitMix

We introduce GaitMix, a soft biometric recognition model integrating multi-view silhouette and 3D pose. Illustrated in the below Figure, our model has four architectural components: silhouette encoder, 3D pose encoder, dual-level fusion module, and estimation module.
![image](https://github.com/jeongdahye3427/ASP-FuseNet/assets/41101841/0049bfba-d9d7-40c7-ae22-09dfeb33a7f2)

We use both OUMVLP and OUMVLP-Mesh datasets in experiments and you can download the dataset here: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html

And, we explain the order of training and testing our model.

### Feature extraction
We extract three types from two modalities (e.g., silhouette and 3D pose)
- GEI features
- HC3D features
- gait featrues
You can extract them by executing FeatureExtraction.py
```
python FeatureExtraction.py
```

### DataSplit
We split the dataset to train, validate, and test sets. You can split dataset by executing DataSplit.py
```
python DataSplit.py
```

### Training
You can train our model by conducting Train.py
```
python Train.py
```

### Testing
If you want to test our model, you need a training weight file, data keys, and extracted features from the mentioned procedure.
You can test our model by conducting Test.py
```
python test.py
```

