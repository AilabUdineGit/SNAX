import pandas as pd
#from spacy.gold import biluo_tags_from_offsets
from spacy.training import  offsets_to_biluo_tags as biluo_tags_from_offsets
import spacy
#from negspacy.negation import Negex
from negspacy.termsets import termset
from utils.interval_merger import IntervalMerger

class SpecExScope():

    def __init__(self):
        nlp = spacy.load("en_core_sci_lg")
        zero_perc, pos_perc = self.build_speculation_dictionary()
        ts = termset("en_clinical_sensitive")
        nlp.add_pipe(
            "negex",
            config={
                "neg_termset": ts.get_patterns(),
                "neg_termset": {
                    "pseudo_negations": [],
                    "preceding_negations": list(zero_perc),
                    "following_negations":list(pos_perc),
                    "termination": []
                }
            }
        )
        #specex = Negex(
        #    nlp,
        #    language = "en_clinical_sensitive",
        #    pseudo_negations=[],
        #    termination=[],
        #    preceding_negations=list(zero_perc),
        #    following_negations=list(pos_perc)
        #)
        #nlp.add_pipe(specex, last=True)
        self.nlp = nlp
        self.err = 0
    
    def biluo_to_iob(self,biluo):
        return [t.replace('U', 'B', 1).replace('L', 'I', 1) for t in biluo]
    
    def iob_to_binary(self,iob):
        return [1 if e[0]=="I" or e[0]=="B" else 0 for e in iob]

    def predict(self, df):
        intervals = list()
        iob_tags_list = list()
        overlpapper = IntervalMerger()
        for i,elem in df.iterrows():
            if type(elem.text) == str:
                doc = self.nlp(elem.text)
                annotations = overlpapper.merge([[e.start_char, e.end_char] for e in doc.ents if e._.negex])
                iob_tags = self.biluo_to_iob(biluo_tags_from_offsets(doc, [(s,e,"") for s,e in annotations]))
                intervals.append(annotations)
                iob_tags_list.append(iob_tags)
            else:
                self.err +=1 
        df['phen_intervals'] = intervals
        df['negation_scope_iob'] = iob_tags_list

        return df


    def build_speculation_dictionary(self):
        cues_df = pd.read_pickle("../datasets/bioscope/bioscope_cues.pickle")
        sent_df = pd.read_pickle("../datasets/bioscope/bioscope_sentences.pickle")
        span_df = pd.read_pickle("../datasets/bioscope/bioscope_final.pickle")

        sent_df.index = sent_df.sentence_id + " " + sent_df.document_id
        span_df.index = span_df.scope_id + " " + span_df.document_id

        cues_df["text"] = ""
        cues_df["span_start"] = 0
        cues_df["span_end"] = 0
        cues_df["span_perc"] = 0

        for idx, row in cues_df.iterrows():
            text = sent_df.loc[row.sentence_id + " "+ row.document_id].sentence[row.start:row.end]
            span_row = span_df.loc[row.scope_id + " "+ row.document_id]
            
            cues_df.at[idx, "text"] = text.lower()
            cues_df.at[idx, "span_start"] = span_row.start
            cues_df.at[idx, "span_end"] = span_row.end
            cues_df.at[idx, "span_perc"] = (row.start - span_row.start) / (span_row.end - span_row.start)

        neg_perc = set()
        zero_perc = set()
        pos_perc = set()
        for perc, df in cues_df[cues_df.type=="speculation"].groupby("span_perc"):
            cues = set(df.text.unique())
            if perc < 0:
                neg_perc |= cues
            elif perc == 0:
                zero_perc |= cues
            else:
                pos_perc |= cues
        return zero_perc, pos_perc