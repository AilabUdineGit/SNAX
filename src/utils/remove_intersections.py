import pandas as pd

def build_baseline_df(row):
    check = lambda elem, intervals : sum([int(elem[0]) in range(int(s),int(e)) or int(elem[1])-1 in range(int(s),int(e)) for s,e in intervals if not (pd.isna(s) or pd.isna(e) )]) > 0
    row['all_predicted_intervals'] = row['predicted_intervals']
    row['predicted_intervals'] = [elem for elem in row.predicted_intervals if not check(elem, row.negation_intervals)]
    return row

def remove_prediction_phen_intersection(df):
    return df.apply(lambda row: build_baseline_df(row),axis=1)
