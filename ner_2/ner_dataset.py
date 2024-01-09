import os
from torch.utils.data import Dataset

class NerDataset(Dataset):

    def __init__(self, sents, sent_tags):
        self.sents = sents
        self.sent_tags = sent_tags

    def __getitem__(self, index):
        return [self.sents[index], self.sent_tags[index]]

    def __len__(self):
        return len(self.sents)

if __name__ == '__main__':
    from corpus_proc import read_corpus
    # 测试读取加载语料
    corpus_dir = os.path.join(os.path.dirname(__file__), 'corpus')
    train_file = os.path.join(corpus_dir, 'example.train')

    sentences,sent_tags,tags = read_corpus(train_file)
    dataset = NerDataset(sentences, sent_tags)

    for sent,tag in dataset:
        print(sent)
        print(tag)
        break
