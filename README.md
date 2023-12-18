# DistilLog
Repository for the paper: Efficient Log-based Anomaly Detection with Knowledge Distillation


**Abstract**: Logs are produced by many systems for troubleshooting purposes. Detecting abnormal events is crucial to maintain regular operation and secure the security of systems. Despite the achievements of deep learning models on anomaly detection, it still remains challenge to apply these deep learning models on edge devices such as IoT devices, due to the limitation of computational resources on these devices. We identify two main problems of adopting these deep learning models in practice, including: (1) they cannot deploy on resource-constrained devices because of the size and speed of large models, and (2) they cannot achieve satisfactory detection accuracy with simple models.  In this work, we proposed a novel lightweight anomaly detection method from system logs, DistilLog to overcome these problems. DistilLog utilizes a pre-trained word2vec model to represent log event templates as semantic vectors, incorporated with the PCA dimensionality reduction algorithm to minimize computational and storage burden. The knowledge distillation technique is applied to reduce the size of the detection model meanwhile maintaining the high detection accuracy. The experimental results show that DistilLog can achieve high F-measures of 0.964 and 0.961 on HDFS and BGL datasets while remain the minimized model size and fastest detection speed. This effectiveness and efficiency demonstrate the ability to deploy the proposed model on resource-constrained systems such as IoT devices.

## Framework
<p align="center"><img src="https://i.ibb.co/Y01YfZs/Framework-of-Distil-Log.png" width="502"><br>An overview of DistilLog</p>


DistilLog consists of the following components:
1. **Preprocessing**: Extracting the log templates using log parser.
2. **Semantic Embedding**: DistilLog extracts the semantic information of log events by applying a pretrained word2vec  model (FastText) then using PCA algorithm for dimensionality reduction.
3. **Classification**: Anomaly detection using a teacher-student architecture-based GRU model with attention. In this step include training student from teacher model via knowledge distillation.

## Requirements
1. Python 3.7+
2. PyTorch 1.11.0
3. cuda 11.3
4. fasttext
5. scikit-learn
6. pandas
7. numpy
8. torchinfo
9. overrides
## Demo
- Train/Test teacher-student Model

See [notebook](demo/DistilLog_on_HDFS.ipynb)
## Data and Models
Datasets: [Data](https://zenodo.org/record/3227177)

Pre-trained models can be found here: [HDFSmodel] (distillog/datasets/HDFS/model.zip) or [BGLmodel] (distillog/datasets/BGL/model.zip)
## Results

Comparison with different model compression techniques (result of testing on CPU)

| Compression   Method |    HDFS   |        |       |                      |    BGL    |        |       |                      | Model size (KB) |  Params |
|:--------------------:|:---------:|:------:|:-----:|:--------------------:|:---------:|:------:|:-----:|:--------------------:|:---------------:|:-------:|
|                      | Precision | Recall |   F1  | Detection   Time (s) | Precision | Recall |   F1  | Detection   Time (s) |                 |         |
|    Original model    |   0.937   |  0.996 | 0.966 |         376.4        |    0.99   |  0.952 | 0.971 |         44.6         |       630       | 160 770 |
|       DistilLog      |   0.932   |  0.999 | 0.964 |         129.2        |   0.979   |  0.944 | 0.961 |          9.4         |        4        |   442   |
|       Prunning       |   0.937   |  0.996 | 0.966 |         371.4        |   0.993   |  0.906 | 0.947 |         39.6         |       630       |   476   |
|     Quantization     |   0.938   |  0.996 | 0.966 |         353.2        |   0.989   |  0.953 | 0.971 |         40.9         |       166       | 160 770 |


## A case study on end devices
A Raspberry Pi 4 device with an ARM Cortex-A72 CPU (1.5GHz) and 4GB RAM is the end device that we chose to deploy DistilLog and LightLog. 

| Dataset |     Metric    | DistilLog | LightLog |
|:-------:|:-------------:|:---------:|:--------:|
|   HDFS  |  T_infer (ms) |   0.252   |   0.493  |
|         | T_detect (ms) |   0.697   |   1.267  |
|         |    CPU Peak   |   45.0%   |   73.6%  |
|         |    RAM Peak   |   22.2%   |   54.4%  |
|   BGL   |  T_infer (ms) |   0.250   |   0.660  |
|         | T_detect (ms) |   0.413   |   1.450  |
|         |    CPU Peak   |   37.5%   |   51.2%  |
|         |    RAM Peak   |   12.2%   |   28.5%  |
