import torch
import argparse
from typing import Dict, Tuple
from pydantic import BaseModel

def parse():
    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    return parser

def initialize(parser):
    # 模型训练参数
    parser.add_argument('--epochs', default=5, help='模型训练迭代数')
    parser.add_argument('--learn_rate', default=1e-5, help='学习率')
    parser.add_argument('--batch_size', default=4, help='训练样本批次数量')
    parser.add_argument('--hidden_size', default=768, help='模型隐藏层大小')
    parser.add_argument('--accum_step', default=4, help='累积梯度步长')
    parser.add_argument('--use_amp', default=True,  help='是否使用混合精度训练')
    # 模型参数
    parser.add_argument('--bert_model', default='bert-base-chinese', help='bert模型名称')
    parser.add_argument('--local_model_dir', default='./bert_model/', help='bert模型本地缓存目录')
    parser.add_argument('--max_position_length', default=512, help='模型输入position最大长度')
    # 语料参数
    parser.add_argument('--train_file', default='./corpus/example.train', help='训练语料文件1')
    parser.add_argument('--dev_file', default='./corpus/example.dev', help='训练语料文件2')
    parser.add_argument('--test_file', default='./corpus/example.test', help='测试语料文件')
    # 模型保存参数
    parser.add_argument('--save_model_dir', default='./saved_model/', help='模型存盘文件夹')
    parser.add_argument('--load_model', default='ner_bertrnn_model_acc_99.32.pth', help='加载的模型存盘文件')

def extension(args):

    class Options(BaseModel):
        epochs: int
        learn_rate: float
        batch_size: int
        hidden_size: int
        accum_step: int
        use_amp: bool
        bert_model: str
        local_model_dir: str
        max_position_length: int
        train_file: str
        dev_file: str
        test_file: str
        save_model_dir: str
        load_model: str
        device: str = 'cpu'
        tags: dict[str,int] = None
        tags_rev: dict[int,str] = None
        entity_pair_ix: dict[str,tuple[int,int]] = None

    options = Options(**args.__dict__)
    # 训练设备
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实体标签
    options.tags = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6, "START": 7, "END": 8}
    options.tags_rev = {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-PER", 4: "I-PER", 5: "B-ORG", 6: "I-ORG", 7: "START", 8: "END"}
    options.entity_pair_ix = {
        'PER':(3,4),
        'ORG':(5,6),
        'LOC':(1,2)
    }
    return options


def get_parser():
    # 初始化参数解析器
    parser = parse()
    # 初始化参数
    initialize(parser)
    # 解析命令行参数
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # 扩展参数
    options = extension(args)
    return options

def main():
    options = get_parser()
    for param, value in options:
        print(param,":", value)
    return options

if __name__ == '__main__':
    main()