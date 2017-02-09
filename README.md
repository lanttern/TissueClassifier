# TissueClassifier

Quick-start instructions:
$ source setup.sh
$ source get_camelyon_data.sh
$ cd camelyon_feature_classifier/
$ jupyter notebook classify_features_camelyon.ipynb
$ cd ../

To use tensorflow/transfer learning --
$ python </path/to/tensorflow>/models/image/imagenet/classify_image.py
$ cp /tmp/imagenet/classify_image_graph_def.pb data/
$ cd camelyon_pool3_classifier/
$ jupyter notebook classify_pool3_camelyon.ipynb
