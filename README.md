<!-- @format -->

# Ultra Fast Lane Detection - CARLA Implementation

## Introduction

[Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) is a cutting-edge project designed to detect lanes on the road in real-time, with a focus on enabling autonomous vehicles to navigate safely and effectively. Our team has adopted this project for use in our university's autonomous car project, building upon the original codebase with our own modifications and optimizations.

We are pleased to share that the results of our work can be found on our [Github page](https://github.com/khann160102/Ultra-Fast-Lane-Detection-CARLA-Implementation), providing an overview of our approach and the performance of our customized [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) model.

In addition to leveraging the capabilities of [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection), we have developed our own tool to generate datasets using CARLA. This tool has proven invaluable in facilitating the training of our models, and we encourage interested parties to explore the source code on our [Github page (in progress)](https://github.com/khann160102/Ultra-Fast-Lane-Detection-CARLA-Implementation) for more information.

## Project Overview

Our project is built upon the foundation of the [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) project. However, we have made significant modifications to the original code and reorganized it into two distinct parts:

1. Training:
   This section is solely dedicated to training the lane detection model. While we have made minor adjustments to the original code, our focus has been on fine-tuning the model for optimal performance.

2. Runtime:
   This portion of the project encompasses everything besides training, including testing and real-world use. We have invested a great deal of effort into developing this codebase to ensure its robustness and versatility.

In addition, we have included numerous scripts in the 'scripts' directory to support various tasks around the project. These scripts cover a range of functions, such as converting or preparing datasets to facilitate model training and testing.

## Requirements

To run our project, you will need the following:

- NVIDIA GPU with CUDA support and latest driver installed
- CUDA Toolkit 1.7 or 1.8
- Python 3.7 to 3.9

We have include a [Installation Tutorial](tutorial/INSTALL.md) file, check it for more information about Python dependencies.

We also recommend using a Linux-based operating system, such as Ubuntu 18.04 or later.

Note that the versions listed above are those that we have tested and confirmed as working with our implementation. While other versions may work, we cannot guarantee their compatibility.

## Usage

To use our project, follow these steps:

1. Clone our repository: git clone https://github.com/yourusername/Ultra-Fast-Lane-Detection.git
2. Install the required dependencies (See [Installation Tutorial](/doc/INSTALL.md) for more information)
3. Download the datasets and models from our Github page.
4. Run the project using the provided scripts.

For more detailed instructions on how to use our project, please see the documentation in the [Usage](/doc/USAGE.md) file.

## Results

Here is a video of our test result on Town10HD dataset from CARLA (Click to open in [YouTube](https://youtu.be/vq4QJsRyKS0)):

[![IMAGE_ALT](https://img.youtube.com/vi/vq4QJsRyKS0/0.jpg)](https://youtu.be/vq4QJsRyKS0)

We plan to test our project with other datasets in the future and will update the results accordingly. We have also provided visualizations of our results in the results directory.

## License

Our project is released under the MIT license. Please see the [LICENSE.md](/LICENSE) file for more information.
