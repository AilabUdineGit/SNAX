from adr_detection.transformer_based_arc import ModelSpans
from utils.remove_intersections import remove_prediction_phen_intersection
from utils.create_iob import create_iob
import os
from utils.utils import save_results, get_neg_or_spec_predictions, check_file_exists, check_pickle_extension
import pandas as pd
from copy import deepcopy

phen_model = ... # name of the negation/speculation detection module

if phen_model not in ["negex","specex","bertneg","bertspec"]:
    raise Exception("You must specify a negation/speculation detection module")

predictions_file = ... # name of the predictions file here
check_pickle_extension(predictions_file)
predictions_file_path = os.path.join("..","datasets","input","model_predictions",predictions_file)
check_file_exists(predictions_file_path)
model_id = predictions_file.split(".")[0]

# original dataset with columns tweet_id and tweet
only_texts_file = ... # name of the predictions file here
check_pickle_extension(only_texts_file)
dataset_texts_path = os.path.join("..","datasets","input","to_evaluate", only_texts_file)
check_file_exists(dataset_texts_path)
dataset_texts = pd.read_pickle(dataset_texts_path)

df = ModelSpans().evaluate(predictions_file_path, dataset_texts)

nn_models_path = os.path.join("..","datasets","output", "neg_spec_models")

df = get_neg_or_spec_predictions(df, phen_model, nn_models_path)
if "negation_scope_iob" in df.columns:
    df = df.drop(columns=['negation_scope_iob'])

df_all = deepcopy(df)
df_cut = remove_prediction_phen_intersection(df)
df_all = create_iob(df_all)
df_cut = create_iob(df_cut)

splits_folder = os.path.join("..","datasets","input", "split_test")
results_path =  os.path.join("..","datasets","output", model_id, phen_model)

os.makedirs(results_path, exist_ok=True)

save_results(df_all, df_cut,
            model_id,
            predictions_file,
            results_path=results_path,
            splits_folder=splits_folder,
            phen_model=phen_model)
