[![pre-commit](https://github.com/milanakuchumova/mlops_hw/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/milanakuchumova/mlops_hw/actions/workflows/pre-commit.yml)

# Image Classification

## About

This project addresses the task of binary classification of images, categorizing them as
"cat" or "dog". The project includes model training, inference (result retrieval), and
experiment monitoring using MLflow.

## Dataset

The dataset consists of images of cats and dogs. For added convenience, the dataset is
automatically downloaded upon executing the `train.py` script.

## Usage

To run the project, follow these steps:

1. Clone this repository
2. Create a new virtual environment for the project and activate it
3. Install all necessary dependencies using the Poetry package manager by running the
   command:

    ```
    poetry install
    ```

4. To initiate the model training process, execute the train.py script. The trained model
   will be saved in the `model/` directory.
5. For inference (obtaining prediction results), you can run the infer.py script. The
   obtained results will be saved in a `.csv` file located in the root directory.

During the model training process, you can also modify the parameters stored in the
`configs/config.yaml` file and track the experiment results in MLflow.

Note: Ensure that the virtual environment is activated before running the specified
commands.

[Link to the dataset and models example](https://github.com/girafe-ai/ml-course/blob/2020_spring/week0_12_CNN/week12_cnn_seminar.ipynb)
