from datetime import datetime
import os
import json
import pandas as pd
from utils.metrics_manager import Metrics
from neg_spec_detection.negex import NegExScope
from neg_spec_detection.specex import SpecExScope
from neg_spec_detection.bertneg import BERTneg
from neg_spec_detection.bertspec import BERTspec

def check_file_exists(path):
    print(path)
    if not os.path.exists(path):
        raise Exception(f"File not available:", path)

def check_pickle_extension(path):
    if not path.endswith('.pkl') and not path.endswith('.pickle'):
        raise Exception("The evaluation file must be a pickle (.pkl, .pickle)")

def get_metrics(df,ids_path):
    ids = [s.replace("\n","") for s in open(ids_path).readlines()]
    tmp = df.loc[ids]
    return Metrics(tmp).return_metrics()

def save_results(df_all, df_cut, model, dataset_to_evaluate, results_path, splits_folder, phen_model):
    date = datetime.now().strftime("[%d/%m/%Y_%H:%M:%S]")

    negade_path     = os.path.join(splits_folder, "negade.id")
    specade_path    = os.path.join(splits_folder, "specade.id")
    noade_path      = os.path.join(splits_folder, "noade.id")
    ade_path        = os.path.join(splits_folder, "ade.id")

    # save dataframe
    save_model_name = model.replace("/","_")
    df_all.to_pickle(os.path.join(results_path, f"all___{save_model_name}.pickle"))
    df_cut.to_pickle(os.path.join(results_path, f"cut___{save_model_name}.pickle"))
    
    
    # save metrics
    if not os.path.exists(os.path.join(results_path, "all_metrics.csv")):
        df_metrics = pd.DataFrame({})
    else:
        df_metrics = pd.read_pickle(os.path.join(results_path, "all_metrics.pickle"))

    all_negade  = get_metrics(df_all, negade_path)
    all_specade = get_metrics(df_all, specade_path)
    all_noade   = get_metrics(df_all, noade_path)
    all_ade     = get_metrics(df_all, ade_path)
    
    cut_negade  = get_metrics(df_cut, negade_path)
    cut_specade = get_metrics(df_cut, specade_path)
    cut_noade   = get_metrics(df_cut, noade_path)
    cut_ade     = get_metrics(df_cut, ade_path)

    metrics_all = Metrics(df_all).return_metrics()
    metrics_cut = Metrics(df_cut).return_metrics()

    df_metrics = df_metrics.append(pd.Series({
        'model': model,
        'dataset': dataset_to_evaluate,
        'metrics_cut': metrics_cut,#metrics_df,
        'metrics_all': metrics_all,#,
        'all_negade':   all_negade,
        'all_specade':  all_specade,
        'all_noade':    all_noade,
        'all_ade':      all_ade,
        'cut_negade':   cut_negade,
        'cut_specade':  cut_specade,
        'cut_noade':    cut_noade,
        'cut_ade':      cut_ade
        #'spurious_cut': metrics_df['partial']['spurious'],
        #'spurious_all': metrics_df_baseline['partial']['spurious']
    }),ignore_index=True)

    df_metrics.to_csv(os.path.join(results_path, "all_metrics.csv"))
    df_metrics.to_pickle(os.path.join(results_path, "all_metrics.pickle"))
    
    cleaned_metrics_path = os.path.join(results_path, "cleaned_metrics.csv")

    if not os.path.exists(cleaned_metrics_path):
        metrics = pd.DataFrame({})
    else:
        metrics = pd.read_pickle(cleaned_metrics_path.replace(".csv",".pickle"))

    f1 = lambda m : (2 * m["partial"]["recall"]*m["partial"]["precision"])/(m["partial"]["recall"]+m["partial"]["precision"]) if (m["partial"]["recall"]+m["partial"]["precision"]) != 0 else 0

    metrics = metrics.append(pd.Series({
        'model': model,
        'dataset': dataset_to_evaluate,
        'phen_model': phen_model,
        'cut_f1': f1(metrics_cut),
        'cut_recall': metrics_cut["partial"]["recall"],
        'cut_precision': metrics_cut["partial"]["precision"],
        
        'all_f1': f1(metrics_all),
        'all_recall': metrics_all["partial"]["recall"],
        'all_precision': metrics_all["partial"]["precision"],
        
        'all_negade_f1':        f1(all_negade),
        'all_negade_recall':    all_negade["partial"]["recall"],
        'all_negade_precision': all_negade["partial"]["precision"],
        'all_negade_spurious':  all_negade["partial"]["spurious"],
        'all_noade_f1':         f1(all_noade),
        'all_noade_recall':     all_noade["partial"]["recall"],
        'all_noade_precision':  all_noade["partial"]["precision"],
        'all_noade_spurious':   all_noade["partial"]["spurious"],
        'all_specade_f1':       f1(all_specade),
        'all_specade_recall':   all_specade["partial"]["recall"],
        'all_specade_precision':all_specade["partial"]["precision"],
        'all_specade_spurious': all_specade["partial"]["spurious"],
        'all_ade_f1':           f1(all_ade),
        'all_ade_recall':       all_ade["partial"]["recall"],
        'all_ade_precision':    all_ade["partial"]["precision"],
        'all_ade_spurious':     all_ade["partial"]["spurious"],

        'cut_negade_f1':        f1(cut_negade),
        'cut_negade_recall':    cut_negade["partial"]["recall"],
        'cut_negade_precision': cut_negade["partial"]["precision"],
        'cut_negade_spurious':  cut_negade["partial"]["spurious"],
        'cut_noade_f1':         f1(cut_noade),
        'cut_noade_recall':     cut_noade["partial"]["recall"],
        'cut_noade_precision':  cut_noade["partial"]["precision"],
        'cut_noade_spurious':   cut_noade["partial"]["spurious"],
        'cut_specade_f1':       f1(cut_specade),
        'cut_specade_recall':   cut_specade["partial"]["recall"],
        'cut_specade_precision':cut_specade["partial"]["precision"],
        'cut_specade_spurious': cut_specade["partial"]["spurious"],
        'cut_ade_f1':           f1(cut_ade),
        'cut_ade_recall':       cut_ade["partial"]["recall"],
        'cut_ade_precision':    cut_ade["partial"]["precision"],
        'cut_ade_spurious':     cut_ade["partial"]["spurious"],
    }),ignore_index=True)
    
    metrics.to_csv(cleaned_metrics_path)
    metrics.to_pickle(cleaned_metrics_path.replace(".csv",".pickle"))
    
    ###

    # save model info
    with open(os.path.join(results_path,"config.json"),"w+") as fp:
        json.dump({
            'date': date,
            'phen_model': phen_model
        },fp)
    print("Results saved in: {}!".format(os.path.join(results_path)))
    ###


def get_evaluated_raw_dataset(path):
    df = pd.read_pickle(path)
    if "doc_id" in df.columns:
        df = df.rename(columns={'doc_id': "tweet_id", "text": "tweet"})
    df = df.set_index("tweet_id")
    return df


def merge_phen_intervals(df_neg, df_spec, row):
    id = row.name
    row["negation_intervals"] = df_neg.loc[id,"negation_intervals"] +  df_spec.loc[id,"negation_intervals"]
    return row


def merge_neg_spec_predictions(df, scope_builder_neg, scope_builder_spec):
    df_neg = scope_builder_neg.predict(df)
    df_spec = scope_builder_spec.predict(df)
    df = df.apply(lambda row: merge_phen_intervals(df_neg, df_spec, row), axis=1)
    return df


def get_neg_spec_models(phen_model, nn_models_path):
    exists_bertneg = not os.path.exists(os.path.join(nn_models_path, "bertneg_bioscope.model"))
    exists_bertspec = not os.path.exists(os.path.join(nn_models_path, "bertspec_bioscope.model"))

    if phen_model == "bertneg":
        return BERTneg(exists_bertneg), None
    if phen_model == "bertspec":
        return BERTspec(exists_bertspec), None
    if phen_model == "negex":
        return NegExScope(), None
    if phen_model == "specex":
        return SpecExScope(), None
    if phen_model == "both_nn":
        return BERTneg(exists_bertneg), BERTspec(exists_bertspec)
    if phen_model == "both_regex":
        return NegExScope(), SpecExScope()
    if phen_model == "bertneg_specex":
        return BERTneg(exists_bertneg), SpecExScope()
    if phen_model == "bertspec_negex":
        return NegExScope(), BERTspec(exists_bertspec)


def get_neg_or_spec_predictions(df, phen_model, nn_models_path):
    neg_model, spec_model = get_neg_spec_models(phen_model, nn_models_path)
    
    if neg_model and spec_model:
        return merge_neg_spec_predictions(df, neg_model, spec_model)
    if neg_model:
        return neg_model.predict(df)
    if spec_model:
        return spec_model.predict(df)
