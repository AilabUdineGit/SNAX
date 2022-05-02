# SNAX
##
The test ids of the tweets are in `datasets/input/split_test` (`S.id`,`N.id` contain only the real samples' ids). In `datasets/input/split_test/train.id` there are all the ids used for the training (except those that are artificial `N` and `S` samples).

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
