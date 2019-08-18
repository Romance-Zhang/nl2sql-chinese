#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:16:08 2019

@author: laiye
"""
device='cpu'
def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu

def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    print('num_out_layers_h',num_out_layers_h)
    print('num_hidden_layers',num_hidden_layers)
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    print('wemb_h',wemb_h.shape)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            print('b1',b1)
            b_pu += 1
            print('b_pu',b_pu)
            for i_nolh in range(num_out_layers_h):
                print('i_nolh',i_nolh)
                i_layer = num_hidden_layers - 1 - i_nolh
                print('i_layer',i_layer)
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                print('st',st)
                print('ed',ed)
                print('encode:',all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:].shape)
                print('wemb:',wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]),         st:ed].shape)
                print('b',b)
                print('i_hds11[0]',i_hds11[0])
                print('i_hds11[1]',i_hds11[1])
                print(all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1]])
                print(all_encoder_layer[i_layer].shape)
                print(all_encoder_layer[i_layer][b].shape)
                print(all_encoder_layer[i_layer][b][i_hds11[0]:i_hds11[1]].shape)
                
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]


    return wemb_h

def generate_inputs(tokenizer, nlu1_tok, hds1):
    hds1 = hds1[0]
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
    print(hds1)
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
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
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
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
        print(nlu_tt1)
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
    print('all_segment_ids',all_input_ids.size())
    print('all_segment_ids',all_input_mask.size())
    print('all_segment_ids',all_segment_ids.size())
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
    #model_bert.to(device)

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

def gen_ques_emb(q_seq,col_seq,model_bert, tokenizer):
    hidden_size = 768
    num_hidden_layers = 12
    bert_path = 'chinese_L-12_H-768_A-12'
    model_bert, tokenizer, bert_config = get_bert(bert_path)
    #nlu_t = [["小明今天去上学"],["张巍好人"]]
    max_seq_length = 255
    #hds = [[['时间']],[['地点']]]
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
               l_n, l_hpu, l_hs, \
               nlu_tt, t_to_tt_idx, tt_to_t_idx = get_bert_output(model_bert, tokenizer, q_seq, col_seq, max_seq_length)

    wemb_n = get_wemb_n(i_nlu, l_n, hidden_size, num_hidden_layers, all_encoder_layer,
                            num_out_layers_n=1)
    print('wemb_n',wemb_n.size())
    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, hidden_size, num_hidden_layers, all_encoder_layer,
                            num_out_layers_h=1)
    print(wemb_h.size())
    x_len =[]
    for i in range(0,len(q_seq)):
        x_len.append(len(q_seq[i][0])+2)

    return wemb_n,x_len