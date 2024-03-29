from spacy.training import offsets_to_biluo_tags as biluo_tags_from_offsets
from spacy.lang.en import English
from utils.interval_merger import IntervalMerger

def fix_ent(doc, ent):
    start,end,type = ent
    start = int(start)
    end = int(end)
    for tok in doc:
        tstart = tok.idx
        tend = tok.idx + len(tok)
        
        if tstart < start < tend:
            start = tstart
            break
        elif start == tstart-1:
            start += 1
            break
            
    for tok in doc:
        tstart = tok.idx
        tend = tok.idx + len(tok)
        
        if tstart < end < tend:
            end = tend
            break
        elif end == tend+1:
            end -= 1
            break
            
    return (start, end, type)


def create_iob(df):
    nlp = English()
    return df.apply(lambda row: build_row_iob(row,nlp),axis=1)


def biluo_to_iob( biluo):
    return [t.replace('U', 'B', 1).replace('L', 'I', 1) for t in biluo]
    
def tagger(doc, annotations):
    annotations = [fix_ent(doc, ann) for ann in annotations]
    overlapper = IntervalMerger()
    annotations = [(s[0],s[1],"ADR") for s in overlapper.merge([[int(o[0]),int(o[1])] for o in annotations ])]
    biluo_tags = biluo_tags_from_offsets(doc, annotations)
    tokens = [str(token) for token in doc]
    return (tokens, biluo_to_iob(biluo_tags))


def build_row_iob(row, nlp):
        
    overlapper = IntervalMerger()

    corr = overlapper.merge([[s,e] for (s,e) in row.correct_intervals])
    pred = overlapper.merge([[s,e] for (s,e) in row.predicted_intervals])
    
    corr = [(s,e, "ADR") for (s,e) in corr]
    pred = [(s,e, "ADR") for (s,e) in pred]

    token,tag_correct = tagger(nlp(row.text), corr)
    _, tag_pred =  tagger(nlp(row.text_normalized), pred)

    row['tokens'] = token
    row['iob_correct'] = tag_correct
    row['iob_prediction'] = tag_pred
    return row
    