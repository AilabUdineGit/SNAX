from tqdm import tqdm
import pandas as pd
import numpy as np

class ModelSpans():

    def from_raw_to_cleaned_data(self, raw_data_path, test_df):
        """
            Take the raw data (models' predictions) with columns:
             - text_id
             - start
             - end 
             - token
            in a dataset with columns:
             - text_id
             - text
             - predicted_ade_intervals
        """
        df = pd.read_pickle(raw_data_path)
        #keys = list(original_dataset.index.values)
        new_df = pd.DataFrame({})
        test_df.set_index("text_id",inplace=True)
        #for key in keys:
        for key in test_df.index.unique():
            key = str(key)
            sentence = test_df.loc[key,"text"]
            correct_intervals = [(int(s),int(e)) for s,e in test_df.loc[key,"correct_intervals"]]
            if key in df.doc_id.values:
                tmp = df[df.doc_id == key]
                preds = list(zip(tmp.start.values,tmp.end.values))
                sentence = tmp.text.values[0]
            else:
                preds = []

            new_df = new_df.append(pd.Series({
                'text_id': str(key),
                'text': sentence,
                'predicted_intervals': preds,
                'correct_intervals': correct_intervals
            }), ignore_index=True)
        
        new_df = new_df.set_index("text_id")
        return new_df

    def evaluate(self,
                 raw_data_path,
                 original_dataset):
        original_dataset.index = original_dataset.index.map(str)
        df = self.from_raw_to_cleaned_data(raw_data_path, original_dataset)
        df['text_normalized'] = df.text
        return df