Torchvision CLI Application. At present only supports pretrained Models.

### **CLI Arguments:**
<pre>
1. --image      : Flag that controls entry to perform operation on an image
2. --video      : Flag that controls entry to perform operation on a video file
3. --realtime   : Flag that controls entry to perform realtime operation
4. --classify   : Flag that controls entry to perform classification
5. --detect     : Flag that controls entry to perform detection
6. --segment    : Flag that controls entry to perform semantic segmentation
7. --all        : Flag that controls whether to detect all faces in the image/video
8. --model-name : Name of the Model (Suported Names given below)
9. --name       : Name of the file (Used when --image or --video is set)
10. --downscale : Used to downscale the video file (Useful for display purposes)
</pre>

Needs one of --image, --video or --realtime and --classify, --detect or --segment

&nbsp;

**Classification Model Names**

<pre>
All images are resized to 256x256 and center cropped to 224x224.

1. alexnet
2. vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
3. resnet18, resnet34, resnet50, resnet101, resnet152
4. wresnet50, wresnet101 (Wide Resnets)
5. resnext50, resnext101 (ResNexts)
6. densenet121, densenet161, densenet169, densenet201
7. mobilenetv2, mobilenetv3_l, mobilenetv3_s
</pre>

&nbsp;

**Detection Model Names**

<pre>
All images are either resized to 914x914, 366x366 or 342x342 and center cropped to 800x800, 320x320 or 300x300 respectively, depending upon the model used.

1. f_resnet50, f_mnet, f_mnet_320 (Faster RCNN: Resnet50, MobilenetV3 Large, MobilenetV3 Large(320x320) Backbones)
2. retinanet (RetinaNet with Resnet50 Backbone)
3. ssd300, ssdlite (SSD300 with VGG16 Backbone, SSDLite with Mobilenet V3 Backbone (320x320))
4. m_resnet50  (Mask Resnet with Resnet 50 Backbone)
</pre>

&nbsp;

**Segmentation Model Names**

<pre>
All images are resized to 592x592 and center cropped to 520x520.

1. fcn_resnet50, fcn_resnet101 (Fully Convolutional Networks)
2. dl_resnet50, dl_resnet101, dl_mobile (Deep Lab V3: Resnet50, Resnet101, MobilenetV3 Large Backbones)
3. lraspp (LR-ASPP with MobilenetV3 Large Backbone)
</pre>
