from tqdm import tqdm
import pandas as pd
import numpy as np

class ModelSpans():
    def __init__(self):
        pass

    def from_raw_to_cleaned_data(self, raw_data_path, test_df):
        """
            Take the raw data (BERT predictions) with columns:
             - tweet_id
             - start
             - end 
             - token
            in a dataset with columns:
             - tweet_id
             - tweet
             - predicted_ade_intervals
        """
        df = pd.read_pickle(raw_data_path)
        #keys = list(original_dataset.index.values)
        new_df = pd.DataFrame({})
        test_df.set_index("tweet_id",inplace=True)
        #for key in keys:
        for key in test_df.index.unique():
            key = str(key)
            sentence = test_df.loc[key,"tweet"]
            correct_intervals = [(int(s),int(e)) for s,e in test_df.loc[key,"correct_intervals"]]
            #sentence = df.loc[key].tweet.values[0]
            if key in df.doc_id.values:
                tmp = df[df.doc_id == key]
                preds = list(zip(tmp.start.values,tmp.end.values))
                sentence = tmp.text.values[0]
            else:
                preds = []

            new_df = new_df.append(pd.Series({
                'tweet_id': str(key),
                'tweet': sentence,
                'predicted_intervals': preds,
                'correct_intervals': correct_intervals
            }), ignore_index=True)
        
        new_df = new_df.set_index("tweet_id")
        return new_df

    def evaluate(self,
                 raw_data_path,
                 original_dataset):
        original_dataset.index = original_dataset.index.map(str)
        df = self.from_raw_to_cleaned_data(raw_data_path, original_dataset)
        #if "correct_intervals" not in df.columns:
        #    df['correct_intervals'] = [[] for _ in range(len(df))]
        #else:
        #    
        # 
        #df = pd.concat([df, original_dataset[['correct_intervals']]], axis=1)
        #df['correct_intervals'] = [k if k != np.nan else [] for k in df['correct_intervals'].values]
        df['tweet_normalized'] = df.tweet
        return df