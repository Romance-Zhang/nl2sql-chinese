import json
from sqlnet.lib.dbengine import DBEngine
import numpy as np
from tqdm import tqdm
import torch
import os
from bert.modeling import BertConfig, BertModel
import bert.tokenization as tokenization
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}
    #所有的标注数据组成一个列表
    for SQL_PATH in sql_paths:
        with open(SQL_PATH, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        print ("Loaded %d data from %s" % (len(sql_data), SQL_PATH))
    #建立一个数据表的对应，id：每个数据表
    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        print ("Loaded %d data from %s" % (len(table_data), TABLE_PATH))
    #如果标注数据的table_id有对应的数据表，就放入ret_sql_data中
    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:    
            ret_sql_data.append(sql)          

    return ret_sql_data, table_data

def load_dataset(toy=False, use_small=False, mode='train'):
    print ("Loading dataset")
    dev_sql, dev_table = load_data('data/val/val.json', 'data/val/val.tables.json', use_small=use_small)
    dev_db = 'data/val/val.db'
    if mode == 'train':
        train_sql, train_table = load_data('data/train/train.json', 'data/train/train.tables.json', use_small=use_small)
        train_db = 'data/train/train.db'
        return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
    elif mode == 'test':
        test_sql, test_table = load_data('data/test/test.json', 'data/test/test.tables.json', use_small=use_small)
        test_db = 'data/test/test.db'
        return dev_sql, dev_table, dev_db, test_sql, test_table, test_db

def conds_ques_wrong(conds,question):
    for i,cond in enumerate(conds):
        if cond[2] not in question:
            return True
    return False
    
def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    vis_seq = []
    sel_num_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        sel_num = len(sql['sql']['sel'])
        sel_num_seq.append(sel_num)
        conds_num = len(sql['sql']['conds'])
        '''
        if conds_ques_wrong(sql['sql']['conds'],sql['question']):
            continue
        '''
        q_seq.append([char for char in sql['question'] if char != ' '])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append(
            (
            len(sql['sql']['agg']),
            sql['sql']['sel'],
            sql['sql']['agg'],
            conds_num,
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds']),
            sql['sql']['cond_conn_op'],
            ))
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq

def to_batch_seq_test(sql_data, table_data, idxes, st, ed):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    table_ids = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append([char for char in sql['question'] if char !=' '])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        raw_seq.append(sql['question'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return q_seq, col_seq, col_num, raw_seq, table_ids

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train(model_bert, tokenizer,model, opt,opt_bert, batch_size, sql_data, table_data):
    model.train()
    perm=np.random.permutation(len(sql_data))
    perm = list(range(len(sql_data)))
    cum_loss = 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq = to_batch_seq(sql_data, table_data, perm, st, ed)
        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conds
        gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(model_bert, tokenizer,q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq, gt_sel_num=gt_sel_num)
        # sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score
        # compute loss
        
        loss = model.loss(score, ans_seq, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        opt.zero_grad()
        opt_bert.zero_grad()
        loss.backward()
        opt.step()
        opt_bert.step()
    return cum_loss / len(sql_data)

def predict_test(model_bert, tokenizer,model, batch_size, sql_data, table_data, output_path):
    model.eval()
    perm = list(range(len(sql_data)))
    fw = open(output_path,'w')
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, col_seq, col_num, raw_q_seq, table_ids = to_batch_seq_test(sql_data, table_data, perm, st, ed)
        score = model.forward(model_bert, tokenizer, q_seq, col_seq, col_num)
        sql_preds = model.gen_query(score, q_seq, col_seq, raw_q_seq)
        for sql_pred in sql_preds:
            sql_pred = eval(str(sql_pred))
            fw.writelines(json.dumps(sql_pred, ensure_ascii=False)+'\n')
            # fw.writelines(json.dumps(sql_pred,ensure_ascii=False).encode('utf-8')+'\n')
    fw.close()

def get_pred(st,ed):
    f = 'data/pre_val.json'
    pred_queries = []
    with open(f, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            if idx>=st and idx<=ed:
                pred_queries.append(sql)
    inf.close()
    return pred_queries

def epoch_acc(model_bert, tokenizer, model, batch_size, sql_data, table_data, db_path, mode_type):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions, new added field
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conditions
        # raw_data: ori question, headers, sql
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data] # original question
        score = model.forward(model_bert, tokenizer, q_seq, col_seq, col_num)
        if mode_type == 1:
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)

        if mode_type == 2:
            pred_queries = get_pred(st,ed)
            # generate predicted format
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt)
        
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        # Execution Accuracy
        for sql_gt, sql_pred, tid in zip(query_gt, pred_queries, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def load_word_emb(file_name):
    print ('Loading word embedding from %s'%file_name)
    f = open(file_name)
    ret = json.load(f)
    f.close()
    # ret = {}
    # with open(file_name, encoding='latin') as inf:
    #     ret = json.load(inf)
    #     for idx, line in enumerate(inf):
    #         info = line.strip().split(' ')
    #         if info[0].lower() not in ret:
    #             ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:16:08 2019

@author: laiye
"""
def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i,i_hds1 in enumerate(i_hds):       
        for (a,b) in i_hds1:
            l_hpu.append(b - a)
    return l_hpu

def generate_inputs(tokenizer, nlu1_tok, hds1):

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = char_tokenize(hds11[0].lower())
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)
    return tokens, segment_ids, i_nlu, i_hds

def char_tokenize(token):
    sub_tokens = []
    for i in range(0,len(token)):
        if token[i]!=' ':
            sub_tokens.append(token[i])
    return sub_tokens

def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            token = token.lower()
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = char_tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))
        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)

        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, nlu_tt1, hds1)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)
        
        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)
        input_ids1 = input_ids1[0:max_seq_length]
        input_mask1 = input_mask1[0:max_seq_length]
        segment_ids1 = segment_ids1[0:max_seq_length]
        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)
    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)
    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)
    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)
    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def get_bert(BERT_PT_PATH):

    bert_config_file = os.path.join(BERT_PT_PATH, 'bert_config.json')
    vocab_file = os.path.join(BERT_PT_PATH, 'vocab.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, 'pytorch_model.bin')


    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    bert_config.print_status()

    model_bert = BertModel(bert_config)

    model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
    print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config

def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n

def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                if i_hds11[1] > 129:
                    continue
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]

    return wemb_h

def gen_ques_emb(model_bert, tokenizer,q_seq,col_seq):
    hidden_size = 768
    num_hidden_layers = 12
    max_seq_length = 130
    #nlu_t = [["小明今天去上学"],["张巍好人"]]
    #hds = [[['时间']],[['地点']]]
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
               l_n, l_hpu, l_hs, \
               nlu_tt, t_to_tt_idx, tt_to_t_idx = get_bert_output(model_bert, tokenizer, q_seq, col_seq, max_seq_length)
    wemb_n = get_wemb_n(i_nlu, l_n, hidden_size, num_hidden_layers, all_encoder_layer,
                            num_out_layers_n=1)
    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, hidden_size, num_hidden_layers, all_encoder_layer,
                            num_out_layers_h=1)
    wemb_n = Variable(wemb_n)
    wemb_h = Variable(wemb_h)
    B = len(q_seq)
    x_len = np.zeros(B, dtype=np.int64)
    for i in range(0,len(q_seq)):        
        x_len[i] = i_nlu[i][1] - i_nlu[i][0]
    
    C = 0
    for i ,i_hds1 in enumerate(i_hds):
        for (a,b) in i_hds1:
            C = C+1
    name_len = np.zeros(C,dtype=np.int64)
            
    m = 0
    num = 0
    for i ,i_hds1 in enumerate(i_hds):
        for (a,b) in i_hds1:
            num = b-a
            name_len[m] = num          
            m = m+1
    col_len = gen_col_batch(col_seq)

    return wemb_n,x_len,wemb_h,name_len, col_len

def gen_col_batch(col_seq):
    col_len = np.zeros(len(col_seq), dtype=np.int64)

    names = []
    for b, one_cols in enumerate(col_seq):
        names = names + one_cols
        col_len[b] = len(one_cols)
    
    return col_len

def str_list_to_batch(str_list):
    B = len(str_list)

    val_len = np.zeros(B, dtype=np.int64)
    for i, one_str in enumerate(str_list):
        val_len[i] = len(one_str[0])
    return val_len

def char_base_to_raw_text(q_seq,col_seq):
    q_seq_new = []
    string = '1'
    for i,char_list in enumerate(q_seq):
        for j in range(0,len(char_list)):
            if char_list[j] != ' ':
                string = string+char_list[j]
        string = string+'2'
        q_tem = []
        q_tem.append(string)
        string = '1'
        q_seq_new.append(q_tem)
    c_seq_new = []
    for i,char_list in enumerate(col_seq):
        c_tem = []
        for j,sub_char_list in enumerate(char_list):
            c_sub_tem = []
            for k in range(0,len(sub_char_list)):
                string = string+sub_char_list[k]
            string = string+'2'
            c_sub_tem.append(string)
            c_tem.append(c_sub_tem)
            string = '1'
        c_seq_new.append(c_tem)
        
    return q_seq_new,c_seq_new
    
    
if __name__ == '__main__':
    bert_path = 'chinese_L-12_H-768_A-12'
    model_bert, tokenizer, bert_config = get_bert(bert_path)
    
    q_seq = [['P', 'E', '(', '对', '应', '2', '0', '1', '8', '.', '1', '0', '.', '3', '1', '收', '盘', '价', '）']]
    col_seq = [[['P', 'E', '(', '对', '应', '2', '0', '1', '8', '.', '1', '0', '.', '3', '1', '收', '盘', '价', '）']]]
    q_seq,col_seq = char_base_to_raw_text(q_seq,col_seq)
    
    '''
    q_seq = [['你好啊，你帮我算一下到底有多少家公司一六年定向增发新股，并且是为了融资收购其他资产的呀']]
    col_seq = [[['时间'],['你好呀']]]
    '''
    wemb_n,x_len,wemb_h,name_len, col_len = gen_ques_emb(model_bert, tokenizer, q_seq, col_seq)
    print('que_size',wemb_n.size())
    print('x_len',x_len)
    print('head_size',wemb_h.size())
    print('name_len',name_len)
    print(col_len)


    
