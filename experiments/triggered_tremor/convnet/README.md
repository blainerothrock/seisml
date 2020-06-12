# Triggered Tremor Classification: ConvNet

Classifying (tremor or no tremor) the Triggered Tremor dataset using convolutional neural network.

## Run
Update `gin.config` with desired parameters.
```shell script
python train.py
```
View results with Tensorboard
```shell script
tensorboard --logdir runs
```
Models will be stored in `./models` (ignored from source control)

## Results
Initial results using the following model with the `20HZ` dataset. Hyperparameters are found in `config.gin`:
```text
------------------------------------------------------------------------------------------
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 64, 78]              --
|    └─Conv1d: 2-1                       [-1, 8, 10001]            24
|    └─AdaptiveAvgPool1d: 2-2            [-1, 8, 10000]            --
|    └─BatchNorm1d: 2-3                  [-1, 8, 10000]            16
|    └─ReLU: 2-4                         [-1, 8, 10000]            --
|    └─Conv1d: 2-5                       [-1, 16, 5001]            272
|    └─AdaptiveAvgPool1d: 2-6            [-1, 16, 5000]            --
|    └─BatchNorm1d: 2-7                  [-1, 16, 5000]            32
|    └─ReLU: 2-8                         [-1, 16, 5000]            --
|    └─Conv1d: 2-9                       [-1, 16, 2501]            528
|    └─AdaptiveAvgPool1d: 2-10           [-1, 16, 2500]            --
|    └─BatchNorm1d: 2-11                 [-1, 16, 2500]            32
|    └─ReLU: 2-12                        [-1, 16, 2500]            --
|    └─Conv1d: 2-13                      [-1, 16, 1251]            528
|    └─AdaptiveAvgPool1d: 2-14           [-1, 16, 1250]            --
|    └─BatchNorm1d: 2-15                 [-1, 16, 1250]            32
|    └─ReLU: 2-16                        [-1, 16, 1250]            --
|    └─Conv1d: 2-17                      [-1, 32, 626]             1,056
|    └─AdaptiveAvgPool1d: 2-18           [-1, 32, 625]             --
|    └─BatchNorm1d: 2-19                 [-1, 32, 625]             64
|    └─ReLU: 2-20                        [-1, 32, 625]             --
|    └─Conv1d: 2-21                      [-1, 32, 313]             2,080
|    └─AdaptiveAvgPool1d: 2-22           [-1, 32, 312]             --
|    └─BatchNorm1d: 2-23                 [-1, 32, 312]             64
|    └─ReLU: 2-24                        [-1, 32, 312]             --
|    └─Conv1d: 2-25                      [-1, 64, 157]             4,160
|    └─AdaptiveAvgPool1d: 2-26           [-1, 64, 156]             --
|    └─BatchNorm1d: 2-27                 [-1, 64, 156]             128
|    └─ReLU: 2-28                        [-1, 64, 156]             --
|    └─Conv1d: 2-29                      [-1, 64, 79]              8,256
|    └─AdaptiveAvgPool1d: 2-30           [-1, 64, 78]              --
|    └─BatchNorm1d: 2-31                 [-1, 64, 78]              128
|    └─ReLU: 2-32                        [-1, 64, 78]              --
├─Sequential: 1-2                        [-1, 2]                   --
|    └─Linear: 2-33                      [-1, 32]                  159,776
|    └─ReLU: 2-34                        [-1, 32]                  --
|    └─Linear: 2-35                      [-1, 16]                  528
|    └─ReLU: 2-36                        [-1, 16]                  --
|    └─Linear: 2-37                      [-1, 2]                   34
==========================================================================================
Total params: 177,738
Trainable params: 177,738
Non-trainable params: 0
------------------------------------------------------------------------------------------
Input size (MB): 0.08
Forward/backward pass size (MB): 4.05
Params size (MB): 0.68
Estimated Total Size (MB): 4.80
------------------------------------------------------------------------------------------
```
![training results](https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_tremor/Screenshot+from+2020-06-12+13-09-50.png)



## Discussion & Next Steps
It is assumed that the above model is overfit, but it is still a good test. The step in this research is the 1.) add new data and 2.) explore more clustering techniques. 
