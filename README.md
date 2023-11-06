# ASP-FuseNet

We introduce ASP-FuseNet, a novel ehanced soft biometric estimation model via silhoutte and 3D pose attention-based fusion networks. Illusterated in below Figure, our model have four of main module: silhouette, 3D pose, bimodal fusion, output.
![model (1)](https://github.com/jeongdahye3427/ASP-FuseNet/assets/41101841/ea762811-aaee-4f0c-834a-621923845f0c)

We use both OUMVLP and OUMVLP-Mesh dataset in experiments and you can download the dataset here: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html
And, we explain the order of training and testing our model.

### Feature extraction
We extract three types from two modalities (e.g., silhouette and 3D pose)
- image feature
- sequntial HC3D features
- gait characteristics featrues
You can extract them by executing FeatureExtraction.py
```
python FeatureExtraction.py
```

### DataSplit
We split dataset to train, validate, test set. You cna split dataset by executing DataSplit.py
```
python DataSplit.py
```

### Training
You can train the our model by conducting Train.py
```
python Train.py
```

### Testing
If you want to test our model, you need training weight file, data keys, extracted features from mentioned procedure.
You can test the our model by conducting Test.py
```
python test.py
```

