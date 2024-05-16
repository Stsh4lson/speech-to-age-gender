# Deprecated
### I do not longer maintain this repository, it's and old university project. I do not stand by this code but I leave it as a reference for someone who may find it usefull.

# Speech-to-AgeGender
Recognition of age and gender based on audio samples of human speech


## Setup
Make sure to use at least python 3.7
```
git clone https://github.com/Stsh4lson/Speech-to-AgeGender
cd Speech-to-AgeGender
pip install -r requirements.txt
```
### Downloading data
Download dataset from https://commonvoice.mozilla.org/en/datasets to have such 
structure in main folder
```
Speech-to-AgeGender
├── ...
├── data
│   ├── en
│   │   ├── clips                  # folder with .mp3 data
│   │   ├── train.tsv              # train samples metadata
│   │   ├── test.tsv               # test samples metadata
```

### Preprocessing data and running model training
Most of preprocessing is done in data_load.py and data_preprocessor modules
```
python saving_to_tfrecord.py
python model_training.py
```

## Sources
dataset from https://commonvoice.mozilla.org/en

ver. en_1932h_2020-06-22
