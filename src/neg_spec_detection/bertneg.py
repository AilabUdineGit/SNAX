from neg_spec_detection.bert_utils.general_bert import GeneralBert

class BERTneg(GeneralBert):
    def __init__(self, train = False):
        self.model_name = "bertneg_bioscope.model"
        self.phenomenon = "negation"
        super().__init__(train)