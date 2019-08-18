import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet

import argparse

def get_opt(model, model_bert, lr, lr_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu to train')
    parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention')
    parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    parser.add_argument('--logdir', type=str, default='', help='Path of save experiment log')
    parser.add_argument('--fine_tune', action='store_true', help='Whether fine_tune')
    args = parser.parse_args()

    n_word=768
    num_hidden_layers = 12
    bert_path = 'code/chinese_L-12_H-768_A-12/'
    if args.toy:
        use_small=True
        gpu=args.gpu
        batch_size=16
    else:
        use_small=False
        gpu=args.gpu
        batch_size=args.bs
    learning_rate = 1e-3
    learning_rate_bert = 1e-3

    # load dataset
    train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=use_small)
    model_bert, tokenizer, bert_config = get_bert(bert_path)
    
    model = SQLNet(N_word=n_word, use_ca=args.ca, gpu=gpu)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    opt, opt_bert = get_opt(model,model_bert,learning_rate,learning_rate_bert,fine_tune=args.fine_tune)
    
    if args.restore:
        model_path= 'saved_model/best_model'
        print ("Loading trained model from %s" % model_path)
        model.load_state_dict(torch.load(model_path))

    # used to record best score of each sub-task
    best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    best_lf, best_lf_idx = 0.0, 0
    best_ex, best_ex_idx = 0.0, 0

    print ("#"*20+"  Star to Train  " + "#"*20)
    for i in range(args.epoch):
        print ('Epoch %d'%(i+1))
        # train on the train dataset
        train_loss = epoch_train(model_bert, tokenizer,model, opt,opt_bert, batch_size, train_sql, train_table)
        # evaluate on the dev dataset
        dev_acc = epoch_acc(model_bert, tokenizer,model, batch_size, dev_sql, dev_table, dev_db,mode_type=1)
        # accuracy of each sub-task
        print ('Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f'%(
            dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6], dev_acc[0][7]))
        # save the best model
        
        if dev_acc[1] > best_lf:
            best_lf = dev_acc[1]
            best_lf_idx = i + 1
            torch.save(model.state_dict(), 'saved_model/best_model')
        if dev_acc[2] > best_ex:
            best_ex = dev_acc[2]
            best_ex_idx = i + 1
        

        # record the best score of each sub-task
        if True:
            if dev_acc[0][0] > best_sn:
                best_sn = dev_acc[0][0]
                best_sn_idx = i+1
            if dev_acc[0][1] > best_sc:
                best_sc = dev_acc[0][1]
                best_sc_idx = i+1
            if dev_acc[0][2] > best_sa:
                best_sa = dev_acc[0][2]
                best_sa_idx = i+1
            if dev_acc[0][3] > best_wn:
                best_wn = dev_acc[0][3]
                best_wn_idx = i+1
            if dev_acc[0][4] > best_wc:
                best_wc = dev_acc[0][4]
                best_wc_idx = i+1
            if dev_acc[0][5] > best_wo:
                best_wo = dev_acc[0][5]
                best_wo_idx = i+1
            if dev_acc[0][6] > best_wv:
                best_wv = dev_acc[0][6]
                best_wv_idx = i+1
            if dev_acc[0][7] > best_wr:
                best_wr = dev_acc[0][7]
                best_wr_idx = i+1
        print ('Train loss = %.3f' % train_loss)
        print ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))
        print ('Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx))
        print ('Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx))
        if (i+1) % 10 == 0:
            print ('Best val acc: %s\nOn epoch individually %s'%(
                    (best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv),
                    (best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx)))
