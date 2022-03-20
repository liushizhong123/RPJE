class Config:
    def __init__(self, trainset):
        self.label = 100
        self.entity_num = trainset.entity_num
        self.relation_num = trainset.relation_num
        self.words_dim = 100
        self.train_embed = True
        self.qa_mode = "LSTM"
        self.hidden_size = 100
        self.num_layer = 2
        self.rnn_dropout = 0.3
        self.rnn_fc_dropout = 0.3





