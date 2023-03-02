<h1 align="center"> SNAX 🥨 </h1>

This repository contains the source code used for the ["Increasing adverse drug events extraction robustness on social media: Case study on negation and speculation"](https://journals.sagepub.com/doi/full/10.1177/15353702221128577) paper.


## Structure
The test ids of the tweets are in `datasets/input/split_test` folder (`S.id`,`N.id` contain only the real samples' ids). In `datasets/input/split_test/train.id` there are all the ids used for the training (except those that are artificial `N` and `S` samples).

In `datasets/input/split_test/neg_spec_samples.csv` you can find the negated/speculated samples. Following the Twitter policy the other tweets cannot be shared.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Execution

1. put the prediction file in `datasets/input/model_predictions/` and add the name's file in `src/main.py` (`predictions_file` variable) 
2. put the file with all the ids and texts evaluated in `datasets/input/to_evaluate/` and add the name's file in `src/main.py` (`only_texts_file` variable) 
3. specify the `phen_model` ("negex","negbert","specex","specbert") in `src/main.py` (even if you don't want to use the models combination strategy)
4. execute `python3 main.py` 

## Cite

```
@article{doi:10.1177/15353702221128577,
author = {Simone Scaboro and Beatrice Portelli and Emmanuele Chersoni and Enrico Santus and Giuseppe Serra},
title ={Increasing adverse drug events extraction robustness on social media: Case study on negation and speculation},
journal = {Experimental Biology and Medicine},
volume = {247},
number = {22},
pages = {2003-2014},
year = {2022},
doi = {10.1177/15353702221128577},
    note ={PMID: 36314865},

URL = { https://doi.org/10.1177/15353702221128577 },
eprint = { https://doi.org/10.1177/15353702221128577 }
```
