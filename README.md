# Char-level CNN model for text classification by PyTorch

This repo contains a PyTorch implementation of a char-level CNN model  for text classification.

The model architecture comes from this paper:<url>https://arxiv.org/pdf/1509.01626.pdf</url>

## Structure of the code

At the root of the project, you will see:

```text
├── pyCharCnn
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── utils # a set of utility functions
├── train_cnn.py
```
## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib

## How to use the code

1. Download the `AG News Topic Classification Dataset` from [coming](url) and place it into the `/pyCharCnn/dataset/raW` directory.
2. Modify configuration information in `pyCharCnn/config/basic_config.py`(the path of data,...).
3. run `python train_cnn.py`.

## Result

coming soon....

### training Figure

coming soon


