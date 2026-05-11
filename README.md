<h1 align="center">Maritime Object Detection from High-Resolution Images</h1>

<p align="center"><strong>Tinsae Ayana Yehuala, Hao Cheng, Ville Lehtola 2025</strong></p>

---

This repository contains the code accompanying the IEEE ITSC 2026 conference paper **Increasing the Efficiency of DETR for Maritime High-Resolution Images**. 

> **Note:** This code was developed and executed by Tinsae Ayana Yehuala as part of his master's thesis at the University of Twente. It is provided as-is to accompany the paper; limited ongoing maintenance should be expected.

The model is a modification of **[RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)**, and incorporates ideas from the following works to create a linearly scalable, faster, and memory-efficient model:

- **[VSSD: Vision Mamba with Non-Causal State Space Duality](https://arxiv.org/abs/2407.18559)**
- **[Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)**
- **[Token Pruning using a Lightweight Background Aware Vision Transformer](https://arxiv.org/abs/2410.09324)**

## Abstract

## Enviroment
Please make sure to use exact version to run the code without error.
- **NVIDIA GPUs**:
    - **cuda compiler Version**: 12.4
    - **gcc and g++ versions**: 11.4.0
    - **python** version: 3.9.21
    - **torch**  version: 2.3.0
- **Other Requiments**: check **requirement.txt** file.  

**Please build all the dependencies in the [third_party](./third_party/) folder from source**

## Model Train, Test and Validation
**Train and validation**
```bash
bash scripts/train.sh <path to config> 
```
**Test**

```bash
bash scripts/test.sh <path to config> <path to model weight>
```

## Models Weights

All the model weights are provided in the [link](https://drive.google.com/file/d/1e0SKsD5HrbHNUgHLAb6PqDNlkM48BUie/view?usp=sharing)

# Repository Structure

- **`config/`**: Contains the configuration file for each model. The configuration files list hyperparameters and directories for datasets, as well as outputs for training and testing.
- **`datasets/`**: Contains the dataloader class and transforms.
- **`models/`**: Contains the implementation of the model architectures.
- **`tools/`**.
- **`thrid_party/`**. dependencies that needs to be built from source. 
- **`scripts/`**. bash scripts for test and training. 
- **`requirements/`**: Contains the required Python packages.

