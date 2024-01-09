import os
import torch
from torch import nn
from torchcrf import CRF
from config import get_parser

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertCRF(nn.Module):

    def __init__(self, bert_model, hidden_dim, target_size):
        super(BertCRF, self).__init__()
        self.bert = bert_model
        self.target_size = target_size
        # 将Bert的输出映射到标签空间
        self.hidden2tag = nn.Linear(bert_model.config.hidden_size, target_size)
        # CRF层
        self.crf = CRF(target_size,batch_first=True)

    def loss(self, out, target, mask):
        return  -1 * self.crf(out, target, mask)

    def decode(self, out, mask):
        return self.crf.decode(out, mask)

    def forward(self, input_ids,token_type_ids,attention_mask):
        # bert模型输出
        # with torch.no_grad():
        #   hidden_states = self.bert(input_ids,token_type_ids,attention_mask)

        # with torch.no_grad():
        embedding_res = self.bert.embeddings(input_ids, token_type_ids, attention_mask)
        hidden_states = self.bert.encoder(embedding_res)

        # bert最后一层的hidden_state
        bert_out = hidden_states.last_hidden_state
        # 推理
        features = self.hidden2tag(bert_out)
        return features  # [batch,seq_len,tag_size]

class BertBiLstmCRF(nn.Module):

    def __init__(self, bert_model, hidden_dim, target_size):
        super(BertBiLstmCRF, self).__init__()
        self.bert = bert_model
        self.target_size = target_size
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim // 2,
                            num_layers=1, bidirectional=True, 
                            batch_first=True)
        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        # CRF层
        self.crf = CRF(target_size,batch_first=True)

    def loss(self, out, target, mask):
        return  -1 * self.crf(out, target, mask)

    def decode(self, out, mask):
        return self.crf.decode(out, mask)

    def forward(self, input_ids,token_type_ids,attention_mask):
        # bert模型输出
        # with torch.no_grad():
        embedding_res = self.bert.embeddings(input_ids, token_type_ids, attention_mask)
        hidden_states = self.bert.encoder(embedding_res)

        # bert最后一层的hidden_state
        bert_out = hidden_states.last_hidden_state
        # lstm所有时间步的输出
        lstm_out, _ = self.lstm(bert_out)
        # 推理
        features = self.hidden2tag(lstm_out)
        return features  # [batch,seq_len,tag_size]

if __name__ == '__main__':
    from model_utils import custom_local_bert, custom_local_bert_tokenizer
    from corpus_proc import read_corpus, generate_dataloader
    from ner_dataset import NerDataset

    # 语料文件
    corpus_dir = os.path.join(os.path.dirname(__file__),'corpus')
    train_file = os.path.join(corpus_dir, 'example.train')
    tags_file = os.path.join(corpus_dir, 'tags.json')

    # # 加载网络模型
    ckpt = 'bert-base-chinese'
    # ckpt = 'chinese-bert-wwm'
    # bert_model, tokenizer = load_transformers_components(ckpt)

    # 加载本地缓存模型目录
    local = os.path.join(os.path.dirname(__file__), 'bert_model/chinese-bert-wwm')
    # 加载定制bert模型
    tokenizer = custom_local_bert_tokenizer(local, max_position=512)
    bert_model = custom_local_bert(local, max_position=512)
    
    opt = get_parser()
    sentences,sent_tags = read_corpus([opt.train_file])
    dataset = NerDataset(sentences, sent_tags)
    data_loader = generate_dataloader(dataset, tokenizer, opt.tags, 4)

    model = BertBiLstmCRF(
        bert_model=bert_model,
        hidden_dim=32,
        target_size=len(opt.tags))
    model.to(device)

    input_data = tokenizer(['测试文本'], return_tensors='pt')
    input_data = (input_data['input_ids'].to(device), input_data['token_type_ids'].to(device), input_data['attention_mask'].to(device))
    writer.add_graph(model, input_data)
    writer.close()

    for train_data in data_loader:
        # 模型训练张量注册到device
        input_ids,token_type_ids,attention_mask,label_ids, label_mask = map(lambda x: train_data[x].to(device) ,train_data)
        # 模型推理
        output = model(input_ids,token_type_ids,attention_mask)
        # 获取模型最后一层的输出
        print(output.shape)
        break