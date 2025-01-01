Files already downloaded and verified
Files already downloaded and verified
CUDA Available? True
cuda
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 64, 16, 16]          18,432
             ReLU-10           [-1, 64, 16, 16]               0
      BatchNorm2d-11           [-1, 64, 16, 16]             128
          Dropout-12           [-1, 64, 16, 16]               0
           Conv2d-13           [-1, 64, 16, 16]             640
           Conv2d-14          [-1, 128, 16, 16]           8,320
             ReLU-15          [-1, 128, 16, 16]               0
      BatchNorm2d-16          [-1, 128, 16, 16]             256
          Dropout-17          [-1, 128, 16, 16]               0
DepthwiseSeparableConv-18          [-1, 128, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,048
           Conv2d-20           [-1, 32, 16, 16]           4,608
             ReLU-21           [-1, 32, 16, 16]               0
      BatchNorm2d-22           [-1, 32, 16, 16]              64
          Dropout-23           [-1, 32, 16, 16]               0
           Conv2d-24           [-1, 64, 16, 16]          18,432
             ReLU-25           [-1, 64, 16, 16]               0
      BatchNorm2d-26           [-1, 64, 16, 16]             128
          Dropout-27           [-1, 64, 16, 16]               0
           Conv2d-28           [-1, 64, 16, 16]             640
           Conv2d-29          [-1, 128, 16, 16]           8,320
             ReLU-30          [-1, 128, 16, 16]               0
      BatchNorm2d-31          [-1, 128, 16, 16]             256
          Dropout-32          [-1, 128, 16, 16]               0
DepthwiseSeparableConv-33          [-1, 128, 16, 16]               0
           Conv2d-34           [-1, 16, 16, 16]           2,048
           Conv2d-35           [-1, 32, 16, 16]           4,608
             ReLU-36           [-1, 32, 16, 16]               0
      BatchNorm2d-37           [-1, 32, 16, 16]              64
          Dropout-38           [-1, 32, 16, 16]               0
           Conv2d-39           [-1, 64, 16, 16]          18,432
             ReLU-40           [-1, 64, 16, 16]               0
      BatchNorm2d-41           [-1, 64, 16, 16]             128
          Dropout-42           [-1, 64, 16, 16]               0
           Conv2d-43           [-1, 64, 16, 16]             640
           Conv2d-44          [-1, 128, 16, 16]           8,320
             ReLU-45          [-1, 128, 16, 16]               0
      BatchNorm2d-46          [-1, 128, 16, 16]             256
          Dropout-47          [-1, 128, 16, 16]               0
DepthwiseSeparableConv-48          [-1, 128, 16, 16]               0
           Conv2d-49           [-1, 16, 16, 16]           2,048
           Conv2d-50           [-1, 32, 16, 16]           4,608
             ReLU-51           [-1, 32, 16, 16]               0
      BatchNorm2d-52           [-1, 32, 16, 16]              64
          Dropout-53           [-1, 32, 16, 16]               0
           Conv2d-54           [-1, 64, 14, 14]          18,432
             ReLU-55           [-1, 64, 14, 14]               0
      BatchNorm2d-56           [-1, 64, 14, 14]             128
          Dropout-57           [-1, 64, 14, 14]               0
           Conv2d-58           [-1, 64, 14, 14]             640
           Conv2d-59          [-1, 128, 14, 14]           8,320
             ReLU-60          [-1, 128, 14, 14]               0
      BatchNorm2d-61          [-1, 128, 14, 14]             256
          Dropout-62          [-1, 128, 14, 14]               0
DepthwiseSeparableConv-63          [-1, 128, 14, 14]               0
        AvgPool2d-64            [-1, 128, 1, 1]               0
           Linear-65                   [-1, 10]           1,290
================================================================
Total params: 137,690
Trainable params: 137,690
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.41
Params size (MB): 0.53
Estimated Total Size (MB): 9.94
----------------------------------------------------------------
</pre>