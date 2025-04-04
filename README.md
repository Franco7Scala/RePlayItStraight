# An Efficient Model Training framework for GreenAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the algorithm "Re-Play it Straight" presented in the research paper "An Efficient Model Training framework for GreenAI". Our proposal aims to reduce the computational and environmental costs of training AI models by strategically pruning the training dataset without compromising performance.

## Key Features

* **Green-AI Focus:** Employs intelligent data pruning to minimize the environmental footprint of AI model training;
* **Re-Play It Straight Algorithm:** Introduces a novel algorithm combining active learning (AL) and repeated random sampling for effective dataset reduction;
* **Comparative Implementations:** Provides scripts for training models with the whole dataset, a pure AL approach, repeated random sampling, and the proposed "Re-Play It Straight" algorithm;
* **Reproducibility:** Facilitates replication of our research results.

## Repository Structure

* **`re_play_it_straight/`:**
    * `main_re_play_it_straight.py`: Implements the proposed "Re-Play It Straight" algorithm for model training.
 
## Requirements

* Python (3.10.6)
* PyTorch (2.2.1)
* Torchvision (0.17.1)
* NumPy (1.26.4)
* CodeCarbon (2.3.4)
* Scikit-learn (1.4.1.post1)
* ptflops (0.7.3)
