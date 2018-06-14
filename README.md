# SQuAD-models
A Tensorflow implementation of models in the SQuAD Leaderboard. Some codes are borrowed from [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net). It's still WIP.


## Dataset
The dataset used for this task is [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/).

## Requirements
  * Python>=2.7
  * NumPy
  * tqdm
  * TensorFlow>=1.5
  * spacy==2.0.9

## TODO's
- [x] Creating training framework
- [x] Add match lstm and R-net
- [ ] Training and testing the model
- [ ] Add and finish QA-net
- [ ] Create Demo
- [ ] Data augmentation by paraphrasing