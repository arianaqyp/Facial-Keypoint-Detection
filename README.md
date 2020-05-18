# Computer Vision: Facial Keypoint Detection

Facial Keypoint Detection is based on the NaimishNet architecture using Haar Cascade on a trained CNN.
This project adapted a modified version of the NaimishNet deep learning architecture to build a facial keypoint detection that takes in any pictures with faces and predicts 68 pairs of keypoints (x, y) on a face.
The model architecture utilises batch normalization as opposed to dropout as seen in the NaimishNet. Xavier uniform initializer is adopted before the fully connected dense layers. The model was trained with batch size 32 and epoch size 150.
The keypoint placements accuracy improved tremendously with an increase in training time (compared to results of batch size 10 and epoch size 14).

### CNN model adapted from NaimeshNet
* Conv1 | in:1 | out:32 | kernel:10 | stride:2
* Relu
* Maxpool | kernel:2
* BatchNorm | num_features:32
* Conv2 | in:32 | out:64 | kernel:5 | stride:2
* Relu
* Maxpool | kernel:2
* BatchNorm | num_features:64
* Conv3 | in:64 | out:128 | kernel:5
* Relu
* Maxpool | kernel:2
* BatchNorm | num_features:128
* Conv4 | in:128 | out:256 | kernel:3
* Relu
* Maxpool | kernel:2
* BatchNorm | num_features:256
* Dense1 | in:256 | out:512
* Dense2 | in:512 | out:136

### NaimishNet
NaimishNet consists of 4 convolutional2d layers, 4 maxpooling2s layers and 3 dense layers, with sandwiched dropout and activation layers.

https://arxiv.org/abs/1710.00977

### Haar Cascade
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
