from neg_spec_detection.bert_utils.general_bert import GeneralBert

class BERTspec(GeneralBert):
    def __init__(self, train = False):
        self.model_name = "bertspec_bioscope.model"
        self.phenomenon = "speculation"
        super(BERTspec,self).__init__(train)