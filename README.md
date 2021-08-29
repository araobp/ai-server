# ai-server

(Work in progress)

## Goal

I have been developing AR applications these days. In some cases, a combination of AR and AI seems to be suitable for work automation.

In this project, I develop image processing argorithms (incl. AI) for AR applications.

```
 [AR app (based on Unity AR Foundation)]-----image---->[AI server]
```

## Construction material recognition on PyTorch

I have been using OpenCV and Tensorflow Lite on Android for some projects at work. But, this time I want to run AI on my PC, with my custom datasets.

This project uses [Detecto (based on PyTorch)](https://github.com/alankbi/detecto) and [its colab demo](https://colab.research.google.com/drive/1ISaTV5F-7b4i2QqtjTa7ToDPQ2k8qEe0) for both training custom datasets and inferencing.

[PTH file for outlet recognition](python/model_weights.pth)

Here is a notebook of object detection test script with the PTH file. 
[Object detection (Jupyter Notebook)](./python/ObjectDetection.ipynb)

## Histgram equalization






