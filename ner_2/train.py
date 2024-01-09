import os
import torch
from tqdm import tqdm
import torchmetrics
from corpus_proc import read_corpus, generate_dataloader
from transformers import AdamW
from ner_dataset import NerDataset
from bert_bilstm_crf import BertCRF, BertBiLstmCRF

from model_utils import custom_local_bert, custom_local_bert_tokenizer, load_ner_model, save_ner_model
from config import get_parser
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

import warnings
# 禁用UserWarning
warnings.filterwarnings("ignore")

def get_dataloader(corpus_files, tags, tokenizer, batch_size=16):
    """
    加载语料文件并通过转换为模型用dataloader
    """
    sentences,sent_tags = read_corpus(corpus_files)
    dataset = NerDataset(sentences, sent_tags)
    data_loader = generate_dataloader(dataset, tokenizer, tags, batch_size)
    return data_loader

def train(opt, model, train_dl, test_dl):
    """
    模型训练方法
    """
    # 模型优化器
    optimizer = AdamW(model.parameters(), lr=opt.learn_rate)

    # training
    for e in range(opt.epochs):
        # evalute(opt, model, test_dl)

        pbar = tqdm(train_dl)
        model.train()
        total_loss = 0

        # 在训练开始时创建一次
        scaler = torch.cuda.amp.GradScaler()

        for i, batch_data in enumerate(pbar):

            # 模型输入
            batch_data = { k:v.to(opt.device) for k,v in batch_data.items()}

            if opt.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(batch_data['input_ids'], batch_data['token_type_ids'], batch_data['attention_mask'])
                    # 计算损失
                    loss = model.loss(output, batch_data['label_ids'], batch_data['label_mask'])
            else:
                # logits
                output = model(batch_data['input_ids'], batch_data['token_type_ids'], batch_data['attention_mask'])
                # 计算损失
                loss = model.loss(output, batch_data['label_ids'], batch_data['label_mask'])

            if opt.use_amp:
                pbar.set_description('Epochs %d/%d loss %f' % (e + 1, opt.epochs, loss.item()))

                global train_loss_cnt
                writer.add_scalar('train loss', loss.item(), train_loss_cnt)
                train_loss_cnt += 1

                # 缩放损失，然后调用backward()
                # 来创建缩放后的梯度
                scaler.scale(loss).backward()

                # 消缩放梯度和调用
                # 或跳过 optimizer.step()
                scaler.step(optimizer)

                # 为下一次迭代更新scaler
                scaler.update()

                optimizer.zero_grad()
            else:
                total_loss += loss * 1 / opt.accum_step

                if i % opt.accum_step == opt.accum_step-1:
                    # 计算模型参数梯度
                    total_loss.backward()
                    # 更新梯度
                    optimizer.step()
                    # 清除累计的梯度值
                    model.zero_grad()

                    pbar.set_description('Epochs %d/%d loss %f'%(e+1, opt.epochs, total_loss.item()))

                    total_loss = 0

        acc = evalute(opt, model, test_dl)
        model.train()
        # 每个epoch后保存模型
        save_ner_model(opt, model, acc)

@torch.no_grad()
def evalute(opt, model, test_dl):

    predict_labels, target_labels = [],[]
    model.eval()
    pbar = tqdm(test_dl)
    for batch_data in pbar:
        batch_data = { k:v.to(opt.device) for k,v in batch_data.items()}
        outputs = model(batch_data['input_ids'], batch_data['token_type_ids'], batch_data['attention_mask'])

        # 解码
        predicted = model.decode(outputs,batch_data['label_mask'])
        pred_tags = [[opt.tags_rev[i] for i in sent[1:-1]] for sent in predicted]
        mask = batch_data['label_mask']
        tag_tags = [[opt.tags_rev[i.item()] for i in sent[mask[j]][1:-1]] for j,sent in enumerate(batch_data['label_ids'])]

        predict_labels.extend(pred_tags)
        target_labels.extend(tag_tags)

        pbar.set_description('collect')


    acc = accuracy_score(target_labels, predict_labels)

    global eval_pr_cnt
    targets_ = [opt.tags[t] for tags in target_labels for t in tags]
    predicts_ = [opt.tags[t] for tags in predict_labels for t in tags]
    writer.add_pr_curve('evaluate pr curve', torch.tensor(targets_), torch.tensor(predicts_), eval_pr_cnt)
    eval_pr_cnt += 1

    print(f'Accuracy of the model on evaluation: {acc * 100:.2f} %')
    print(classification_report(target_labels, predict_labels))
    return acc * 100

if __name__ == '__main__':

    # loss跟踪计数器
    train_loss_cnt = 0
    # pr评估计数器
    eval_pr_cnt = 0

    # 加载模型相关参数
    opt = get_parser()
    # 本地模型缓存目录
    local = os.path.abspath(os.path.join(opt.local_model_dir, opt.bert_model))
    # 加载定制bert模型
    bert_model = custom_local_bert(local, max_position=opt.max_position_length)
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)

    # 模型语料文件目录
    train_file = os.path.abspath(opt.train_file)
    dev_file = os.path.abspath(opt.dev_file)
    test_file = os.path.abspath(opt.test_file)

    # 模型训练语料
    train_dl = get_dataloader([train_file,dev_file], opt.tags, tokenizer, batch_size=opt.batch_size)
    test_dl = get_dataloader([test_file], opt.tags, tokenizer)

    # BiLSTMCRF模型
    model = BertBiLstmCRF(
        bert_model=bert_model,
        hidden_dim=opt.hidden_size,
        target_size=len(opt.tags))

    # 连续训练，加载之前存盘的模型
    if os.path.exists(os.path.join(opt.save_model_dir, opt.load_model)):
            model = load_ner_model(opt, BertBiLstmCRF)

    model.to(opt.device)
    # 模型训练
    train(opt, model, train_dl, test_dl)