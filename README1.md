## Challenge 
The **Advanced** task covers all problems in **Beginners** task. Therefore, this project contains solutions for the **Advanced** challenge. 


## README.md 
This page explains the overall structure of the project. The detail information about solutions of given challenge is in Documentation.md under **classification** folder. 

The project consists of six items:

1. **Given source code - cifar10_pt.py**
2. **Fixed source code - cifar10_fixed.py** 
3. **Improved source code - cifar10_improved.py**
4. **classification** folder contains the restructured solution as **cifar10.py** that follows the Object-Oriented programming style and PEP8 compliant.
5. **mAP (Mean Average Precision) metric** is implemented in **metric.py** and **cifar10_mAP.py** in classification folder shows how to use mAP metric.
6. **Documentation.md** presents how the performance of the model improved and the summary of experiment results


### Runtime Error Fix
The original source code cifar10_pt.py contained one runtime error, one variable mistake and unnecessary imported library. 
The following three items are fixed to run cifar10_pt.py correctly.

1. In line 42, the shape of data structure (matrix) inbetween convolutional layer and linear layer was not matching. 
To match the shape of data structure. [2048, 1024] become [4096, 1024].  


```
weights = [
    [3, 32, 3],
    [32, 32, 3],
    [32, 64, 3],
    [64, 64, 3],
    [64, 128, 3],
    [128, 256, 3],
    [4096, 1024],  # Error [2048, 1024],
    [1024, 512],
    [512, 256],
    [256, 128],
    [128, 64],
    [64, output_classes]
]
```

2. The wrong variable name was used. (I guess it was a typo)
```
     with torch.no_grad():
            for steps, data, in enumerate(test_ds_loader):
                images, labels = data
                pred = model(images)

                _, predicted = torch.max(pred.data, 1)  # Error torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
```

3. In line 4, Tensorflow library is not necessary for PyTorch code
```
from tensorflow.python.ops.nn_ops import dropout # NO NEED
```

## Requirements 
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the solution, run this command:

```train
python cifar10_[version].py
```

|    name      | description   |
|--------------|---------------|
| cifar10_pt.py| original source code |
| cifar10_fixed.py | fixed runtime error, but it keeps the original code structure |
| cifar10_improved.py| fixed runtime error and improved the training speed and model accuracy, but it keeps the original code structure |
| /classification/cifar10.py | improved model with OOP based structured code |
| /classification/cifar10_mAP.py   | improved model with OOP based structured code and measure the mAP (Mean Average Precision) metric |

## Evaluation
There are two model evaluation metrics in this solution.
1. Accuracy : It measures how many prediction are correct with respect to label, among all predictions. 
Accuracy Score = (TP + TN) / (TP + TN + FP + FN)

3. Mean Average Precision (mAP) : It measures the average of of each classes' average precision (AP). First, it calculates the average precision (AP) of each class using precision and recall. Second, it calculates the average of all classes' AP.   

