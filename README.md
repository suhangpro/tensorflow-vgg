# Tensorflow VGG16 and VGG19

This is a modified version of [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). 

New features: 

 - VGG-16 and VGG-19 model files will be downloaded from a server if not found on disk. 
 - A unified interface is provided: you can now use `vgg.VggVd(model='vgg16')` and `vgg.VggVd(model='vgg19')`
 - Support building only part of the vgg networks with the `layer_range` option of `build()`
 - Optional random initialization for the last layer for the use in fine-tuning (set the `num_classes` option of `build()`)
 - Useful summaries can be added to all activations (turn on the `summary` option of `build()`)
 - In training mode (when `train` option of `build()` is set to `True`), all network parameters are variables instead of constants. 
 - Weight decay can be added to all filters (set `weight_decay` option of `build()` to non-zero float values)
 - Major layers can be accessed using the vgg object, i.e. `vgg.conv1`...`vgg.conv5`, `vgg.pool1`...`vgg.pool5`, `vgg.fc6`...`vgg.fc8`, `vgg.prob`, as well as the associated parameters: `vgg.conv1_params`...`vgg.conv5_params`, `vgg.fc6_params`...`vgg.fc8_params`. 

Other changes: 

 - Input should be in the range of [0.0, 255.0] instead of [0.0, 1.0]

##Usage
Use this to build the VGG object
```
net = vgg.VggVd(model='vgg19')
net.build(images)
```
or
```
net = vgg.VggVd(model='vgg16')
net.build(images)
```
The `images` is a tensor with shape `[None, 224, 224, 3]`. 
Create a network for fine-tuning: 
```
net = vgg.VggVd()
prob, params = net.build(images, num_classes=NUM_CLASSES, use_variable=True, summary=True, weight_decay=0.001)
```
>Trick: the tensor can be a placeholder, a variable or even a constant.

`test_vgg.py` contains the sample usage.
