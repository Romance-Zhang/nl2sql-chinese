import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
    parser.add_argument('--toy', action='store_true', help='Small batchsize for fast debugging.')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention.')
    parser.add_argument('--train_emb', action='store_true', help='Use trained word embedding for SQLNet.')
    parser.add_argument('--output_dir', type=str, default='', help='Output path of prediction result')
    parser.add_argument('--mode_type', type=int, default=1, help='which mode')
    args = parser.parse_args()

    n_word=768
    bert_path = 'code/chinese_L-12_H-768_A-12'
    if args.toy:
        use_small=True
        gpu=args.gpu
        batch_size=16
    else:
        use_small=False
        gpu=args.gpu
        batch_size=64
    

    dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')
    #train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=use_small, mode='test')
    model_bert, tokenizer, bert_config = get_bert(bert_path)
    model = SQLNet(N_word=n_word, use_ca=args.ca, gpu=gpu, trainable_emb=args.train_emb)

    model_path = 'saved_model/best_model'
    print ("Loading from %s" % model_path)
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    print ("Loaded model from %s" % model_path)
    dev_acc = epoch_acc(model_bert, tokenizer,model, batch_size, dev_sql, dev_table, dev_db, args.mode_type)
    print ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))

    print ("Start to predict test set")
    predict_test(model_bert, tokenizer, model, batch_size, test_sql, test_table, args.output_dir)
    print ("Output path of prediction result is %s" % args.output_dir)
