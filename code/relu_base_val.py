# -!- coding: utf-8 -!-
import numpy as np
import json
import re
import jieba
import pandas as pd
import copy

def get_date(question):
    a = re.findall('\d+年+\d+月+\d+[日/号]',question)
    #######把问句中的年月日替换为-连接的字符串，方便与数据库中数据匹配
    if a:
        tem = re.sub('年','-',a[0])
        tem = re.sub('月','-',tem)
        question = re.sub(a[0],tem,question)
    else:
        a = re.findall('\d+年+\d+月',question)
        if a:
            tem = re.sub('年','-',a[0])
            question = re.sub(a[0],tem,question)
    ###顺便替换‘两’
    question = re.sub('两','二',question)
    return question
            
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def is_alabo_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

#选出所有问句中的数字来
def chinese_to_num(word):    
    #####1.拿到包含汉字数字的连续字符串
    ########  x百做特殊处理
    valid_list = ['零','一','二','三','四','五','六','七','八','九','十','点']
    chi_list = []
    tem = ''
    for i in range(0,len(word)):
        if word[i] in valid_list:
            tem += word[i]
            if i == len(word)-1:
                chi_list.append(tem)
            continue
        if tem != '':
            chi_list.append(tem)
            tem = ''
        
    #####2.将汉字转换为数字  
    chi_str = ''
    number = ''
    if len(chi_list) > 0:
        chi_str = chi_list[0]
    else:
        return word
    if len(chi_str) == 1:
        number = str(valid_list.index(chi_str))

    elif '点' in chi_str:
        number = ''
        if chi_str.index('点') == 1:
            for i in range(0,len(chi_str)):
                if chi_str[i] == '点':
                    number += '.'
                else:
                    if chi_str[i]!='十':
                        number += str(valid_list.index(chi_str[i]))
                    else:
                        if i != len(chi_str)-1 and chi_str[i+1] == '点':
                            number += '10'
                        else:
                            number += '1'
        elif chi_str.index('点') == 2:
            if chi_str[0] == '十':
                number = '1' + str(valid_list.index(chi_str[1]))
            elif chi_str[1] == '十':
                number = str(valid_list.index(chi_str[0])) + '0'
            for i in range(2,len(chi_str)):
                if chi_str[i] == '点':
                    number += '.'
                else:
                    number += str(valid_list.index(chi_str[i]))
                
        elif chi_str.index('点') == 3:
            
            for i in range(0,len(chi_str)):
                if chi_str[i] == '点':
                    number += '.'
                    continue
                if chi_str[i] != '十':
                    number += str(valid_list.index(chi_str[i]))
                
    elif len(chi_str) == 2:
        if chi_str[0] == '十':
            number = '1' + str(valid_list.index(chi_str[1]))
        elif chi_str[1] == '十':
            number = str(valid_list.index(chi_str[0])) + '0'
    
    else:
        number = ''
        for i in range(0,len(chi_str)):
            if chi_str[i] != '十':
                number += str(valid_list.index(chi_str[i]))
                
    ############3.百做特殊处理
    if '百' in word and word[word.index('百')-1] in valid_list:
        judge = word.index('百')+1 if word.index('百')+1<len(word) else word.index('百')
        if word[judge] == '分':
            pass
        else:
            re_str = word[word.index('百')-1] + '百'
            re_num = str(valid_list.index(word[word.index('百')-1]))+'00'
            word = re.sub(re_str,re_num,word)
    ######################3.字符串中进行替换   
    if chi_str in word:
        word = re.sub(chi_str,number,word)
    return word    

def find_lcseque(s1, s2):   
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果  
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]   
    # d用来记录转移方向  
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]   

    for p1 in range(len(s1)):   
        for p2 in range(len(s2)):   
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1  
                m[p1+1][p2+1] = m[p1][p2]+1  
                d[p1+1][p2+1] = 'ok'            
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向  
                m[p1+1][p2+1] = m[p1+1][p2]   
                d[p1+1][p2+1] = 'left'            
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up  
                m[p1+1][p2+1] = m[p1][p2+1]     
                d[p1+1][p2+1] = 'up'           
    (p1, p2) = (len(s1), len(s2))   
    s = []   
    while m[p1][p2]:    #不为None时  
        c = d[p1][p2]  
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个  
            s.append(s1[p1-1])  
            p1-=1  
            p2-=1   
        if c =='left':  #根据标记，向左找下一个  
            p2 -= 1  
        if c == 'up':   #根据标记，向上找下一个  
            p1 -= 1  
    s.reverse()   
    out = ''.join(s)
    return out,len(out)

def getNumofCommonSubstr(str1, str2):
  
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)] # 多一位
    maxNum = 0   # 最长匹配长度
    p = 0    # 匹配的起始位
  
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i+1][j+1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p-maxNum:p], maxNum

def match_chinese_value(rows,question):
    ########################匹配进行字符串的选择策略是：
    ##############对rank1_str 1.比例最大的优先  2.比例相同的字符长的优先  
    ##############对rank2_str 1.比例最大的优先  2.比例相同的不重复优先（不在rank1_str中）  3.比例相同的
    rank1 ,rank2 ,header1 ,header2= 0 ,0 ,0 ,0
    rank1_str ,rank2_str = '' ,''

    for i in range(0,len(rows)):
        for j in range(0,len(rows[0])):
            rows[i][j] = str(rows[i][j])
            if rows[i][j] == rank1_str or rows[i][j] == rank2_str: 
                continue
            _,num1 = find_lcseque(question,rows[i][j])
            num = num1/len(rows[i][j])
            ###########做一个长度的优化，长度较短的减少一定的数值
            if len(rows[i][j]) == 1:
                num = num - 0.4
            if len(rows[i][j]) == 2:
                num = num - 0.15
            if len(rows[i][j]) == 3:
                num = num - 0.12
            if len(rows[i][j]) == 4:
                num = num - 0.1
                
            if num > rank1:                   ####比例最大的优先
                rank2,rank2_str,header2 = rank1,rank1_str,header1     
                rank1,rank1_str,header1 = num,rows[i][j],j
            elif num == rank1:  
                if len(rows[i][j]) > len(rank1_str):    ########比例相同的字符长的优先
                    rank2,rank2_str,header2 = rank1,rank1_str,header1  
                    rank1,rank1_str,header1 = num,rows[i][j],j
                else:
                    if num > rank2:
                        rank2,rank2_str,header2 = num,rows[i][j],j
                    else:   ########### num == rank2
                        if len(rows[i][j]) > len(rank2_str):
                            rank2,rank2_str,header2 = num,rows[i][j],j
                                                            
            elif num > rank2:
                rank2,rank2_str,header2 = num,rows[i][j],j
            
            elif num == rank2:                ########比例相同的不重复优先 
                if rank2_str in rank1_str:
                    if rows[i][j] not in rank1_str :
                        rank2,rank2_str,header2 = num,rows[i][j],j
                    else:
                        if len(rows[i][j]) > len(rank2_str):
                            rank2,rank2_str,header2 = num,rows[i][j],j
                else:
                    if rows[i][j] not in rank1_str :
                        if len(rows[i][j]) > len(rank2_str):
                            rank2,rank2_str,header2 = num,rows[i][j],j
    
    return rank1_str,rank2_str,header1,header2

def num_column(word,question,headers):
    st = question.index(word)- 12 if question.index(word)- 12 > 0 else 0
    ed = question.index(word)
    question = question[st:ed]
    rows = [headers]
    _,_,header1,header2 = match_chinese_value(rows,question)
    #############把question和数据表中rows中的数值进行比对得到匹配度最高的header1和header2
    return header1,header2

def get_headers(table_id):
    f = '../data/val/val.tables.json'   
    with open(f, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            my_id = sql['id']
            if my_id == table_id:
                headers = sql['header']
                inf.close()
                return headers
                   
def get_number(re_list):
    new_list = []
    for word_piece in re_list:
        if '%' in word_piece:
            word_piece = re.sub('%','',word_piece)
        word_piece = chinese_to_num(word_piece)
        cut_list = jieba.lcut(word_piece)
        for i,cut in enumerate(cut_list):
            if is_number(cut):
                if i>0:
                    if cut_list[i-1] == '-':
                        cut = '-'+cut 
                if cut not in ['万','千','百','亿']:
                    new_list.append(cut)
                    break 
    return new_list

def get_number_spe(re_list):
    new_list = []
    for word_piece in re_list:
        if '%' in word_piece:
            word_piece = re.sub('%','',word_piece)
        word_piece = chinese_to_num(word_piece)
        cut_list = jieba.lcut(word_piece)
        cut_list = list(reversed(cut_list))
        for i,cut in enumerate(cut_list):
            if is_number(cut):
                if i>0:
                    if cut_list[i-1] == '-':
                        cut = '-'+cut
                if cut not in ['万','千','百','亿']:
                    new_list.append(cut)
                    break            
    return new_list

def get_piece(word,question,headers):
    re_list = []
    header1_list = []
    header2_list = []
    if word in question:
#############用之前的字符判断列#############
        if question[question.index(word)-1]!='不' and question[question.index(word)-1]!='未'and question[question.index(word) - 2:question.index(word)]!='没有' and question[question.index(word)-1]!='没':
            st = question.index(word)+len(word)
            ed = st+8 if st+8<len(question) else len(question)
            re_sub_piece = question[st:ed]
            re_list.append(re_sub_piece)
            header1,header2 = num_column(word,question,headers)
            header1_list.append(header1)
            header2_list.append(header2)
            if question[question.index(word) - 1] == '都' or question[question.index(word) - 2:question.index(word)] == '都是' or question[question.index(word) - 2:question.index(word)] == '都会':
                re_list.append(re_sub_piece)
            if word in question[st:]:
                a,b,c = get_piece(word,question[st:],headers)
                if len(a)>0:
                    re_list.append(a[0]) 
                    header1,header2 = num_column(word,question[st:],headers)
                    header1_list.append(header1)
                    header2_list.append(header2)
                    
    return re_list,header1_list,header2_list


def get_spe_piece(word,question,headers):
    re_list = []
    header1_list = []
    header2_list = []
    if word in question:
        ed = question.index(word)
        st = ed-10 if ed-10>=0 else 0
        re_sub_piece = question[st:ed]
        re_list.append(re_sub_piece)
        header1,header2 = num_column(word,question,headers)
        header1_list.append(header1)
        header2_list.append(header2)
        if word in question[ed+len(word):]:
            a,b,c = get_spe_piece(word,question[ed+len(word):],headers)
            if len(a) !=0:
                re_list.append(a[0])
                header1,header2 = num_column(word,question[ed+len(word):],headers)
                header1_list.append(header1)
                header2_list.append(header2)
        
    return re_list,header1_list,header2_list

def re_val():
    more_list = ['多于','大于','高于','超过','超出','突破','不低于','不小于','占比过','达到','超','破']
    more_spe_list = ['以上']
    less_list = ['少于','小于','低于','不足','不高于','不超过','不大于','未达到','没有达到','没有超过','没破','不到']
    less_spe_list = ['以下']
    f = '../data/val/val.json'
    bb_more_list ,bb_less_list ,bb_more_spe_list ,bb_less_spe_list = [],[],[],[]
    hhh_more_list1 ,hhh_less_list1 ,hhh_more_spe_list1 ,hhh_less_spe_list1 = [],[],[],[]
    hhh_more_list2 ,hhh_less_list2 ,hhh_more_spe_list2 ,hhh_less_spe_list2 = [],[],[],[]
    
    with open(f, encoding='utf-8') as inf:        
        for idx, line in enumerate(inf):
            print(idx)
            sql = json.loads(line.strip())            
            question = sql['question']
            table_id = sql['table_id']
            headers = get_headers(table_id)
            re_more_list ,re_less_list ,re_more_spe_list ,re_less_spe_list = [],[],[],[]
            hd_more_list1 ,hd_less_list1 ,hd_more_spe_list1 ,hd_less_spe_list1 = [],[],[],[]
            hd_more_list2 ,hd_less_list2 ,hd_more_spe_list2 ,hd_less_spe_list2 = [],[],[],[]
               
            for word in more_list:
                re_sub_list,hd_sub_list1,hd_sub_list2 = get_piece(word,question,headers)
                if re_sub_list:
                    re_more_list.extend(re_sub_list)
                    hd_more_list1.extend(hd_sub_list1)
                    hd_more_list2.extend(hd_sub_list2)

            for word in less_list:
                re_sub_list,hd_sub_list1,hd_sub_list2 = get_piece(word,question,headers)
                if re_sub_list:
                    re_less_list.extend(re_sub_list)
                    hd_less_list1.extend(hd_sub_list1)
                    hd_less_list2.extend(hd_sub_list2)
            
            for word in more_spe_list:         #################如果以上之前出现了别的词语，则跳过
                flag = True
                if word in question:
                    sub_st = question.index(word)-9 if question.index(word)-9 > 0 else 0
                    new_ques = question[sub_st:question.index(word)] 
                    for sub_more in more_list:
                        if sub_more in new_ques:
                            flag = False
                            break
                    if flag == False:
                        continue
                    re_sub_list,hd_sub_list1,hd_sub_list2 = get_spe_piece(word,question,headers)
                    if re_sub_list:
                        re_more_spe_list.extend(re_sub_list)
                        hd_more_spe_list1.extend(hd_sub_list1)
                        hd_more_spe_list2.extend(hd_sub_list2)
                    
            for word in less_spe_list:
                flag = True
                if word in question:
                    sub_st = question.index(word)-9 if question.index(word)-9 > 0 else 0
                    new_ques = question[sub_st:question.index(word)]
                    for sub_less in less_list:
                        if sub_less in new_ques:
                            flag = False
                            break
                    if flag == False:
                        continue
                    re_sub_list,hd_sub_list1,hd_sub_list2 = get_spe_piece(word,question,headers)
                    if re_sub_list:
                        re_less_spe_list.extend(re_sub_list)
                        hd_less_spe_list1.extend(hd_sub_list1)
                        hd_less_spe_list2.extend(hd_sub_list2)
            
            re_more_list = get_number(re_more_list)
            re_less_list = get_number(re_less_list)
            re_more_spe_list = get_number_spe(re_more_spe_list)
            re_less_spe_list = get_number_spe(re_less_spe_list)
            
            bb_more_list.append(re_more_list)
            bb_less_list.append(re_less_list)
            bb_more_spe_list.append(re_more_spe_list)
            bb_less_spe_list.append(re_less_spe_list)
            
            hhh_more_list1.append(hd_more_list1)
            hhh_less_list1.append(hd_less_list1)
            hhh_more_spe_list1.append(hd_more_spe_list1)
            hhh_less_spe_list1.append(hd_less_spe_list1)
            
            hhh_more_list2.append(hd_more_list2)
            hhh_less_list2.append(hd_less_list2)
            hhh_more_spe_list2.append(hd_more_spe_list2)
            hhh_less_spe_list2.append(hd_less_spe_list2)
            
    inf.close()
    re_list = [bb_more_list,bb_less_list,bb_more_spe_list,bb_less_spe_list,hhh_more_list1, hhh_less_list1,hhh_more_spe_list1,hhh_less_spe_list1,hhh_more_list2, hhh_less_list2,hhh_more_spe_list2,hhh_less_spe_list2]
    return re_list

def judge_special_num(c,rows,header):
    for i in range(0,len(rows)):
        for j in range(0,len(rows[i])):
            if float(c) == rows[i][j]:
                return str(rows[i][j]),j
    return '',0

def get_special_num(question,rows,headers):
    rank_str = ''
    header = 0
    more_list = ['多于','大于','高于','超过','超出','突破','不低于','不小于','占比过','达到','超','破']
    more_spe_list = ['以上']
    less_list = ['少于','小于','低于','不足','不高于','不超过','不大于','未达到','没有达到','没有超过','没破','不到']
    less_spe_list = ['以下']
    a_list = more_list
    a_list.extend(more_spe_list)
    a_list.extend(less_list)
    a_list.extend(less_spe_list)
    for sub in a_list:
#        if sub in question:
            return rank_str,header
    c = jieba.lcut(question)
    while '-' in c:
        pos = c.index('-')
        if pos > 0 and is_alabo_number(c[pos-1]):
            del c[pos-1]
        pos = c.index('-')
        if pos < len(c)-1 and is_alabo_number(c[pos+1]):
            del c[pos+1]
        pos = c.index('-')
        del c[pos]
    for i in range(0,len(c)):
        if is_alabo_number(c[i]):
            if float(c[i])>2000 and float(c[i])<2021:
                continue
            else:
                rank_str,header = judge_special_num(c[i],rows,header)             
                if rank_str:
                   return rank_str,header
    return rank_str,header      
        
def column_exact_match(question,origin_rows,headers):
    rows = copy.deepcopy(origin_rows)
    question = get_date(question)
    for i ,row in enumerate(rows):
        for j,sub_row in enumerate(row):
            sub_row = str(sub_row)
            rows[i][j] = str(rows[i][j])
            sub_row = re.sub(',','',sub_row)
            sub_row = re.sub('%','',sub_row)
            if 'e+' in sub_row:
                continue
            if is_number(str(sub_row)):
                rows[i][j] = '000000000000'          
    rank_list = []
    header_list = []
    ban_list = ['(','[','{','（','）',')','“','”']
    rank1_str,_,header1,_ = match_chinese_value(rows,question)
    common_str,_ = getNumofCommonSubstr(question,rank1_str)
    #common_str,_ = find_lcseque(question,rank1_str)
    for char in ban_list:
        if char in common_str:
            question = question.replace(common_str,'')
            break
        else:
            question = re.sub(common_str,'',question)   
    
    rank2_str,_,header2,_ = match_chinese_value(rows,question)
    common_str,_ = getNumofCommonSubstr(question,rank2_str)
    #common_str,_ = find_lcseque(question,rank2_str)
    for char in ban_list:
        if char in common_str:
            question = question.replace(common_str,'')
            break
        else:
            question = re.sub(common_str,'',question)
            
    rank3_str,_,header3,_ = match_chinese_value(rows,question)

    if rank2_str == rank1_str:
        rank2_str = rank3_str
        header2 = header3            
       
    rank_str,header = get_special_num(question,origin_rows,headers)
    if rank_str:
        rank3_str = rank_str
        header3 = header     
            
    rank_list.append(rank1_str)
    rank_list.append(rank2_str)
    rank_list.append(rank3_str)
    header_list.append(header1)
    header_list.append(header2)
    header_list.append(header3)
    
    return rank_list,header_list

def column_match(question,table_id):
    f = '../data/val/val.tables.json'
    with open(f, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            rows = sql['rows']
            headers = sql['header']
            if table_id == sql['id']:
                columns,header = column_exact_match(question,rows,headers)
                inf.close()
                return columns,header

def get_input_dict(i):
    f = '../data/best_val.json'
    inf = open(f,encoding='utf-8')
    sql = json.loads(inf.readlines()[i])
    inf.close()
    return sql


def first_step():
    f = '../data/val/val.json'
    out_dict = {}
    out_dict_list = []
    ############1.用于预测where_column和where_value部分，并根据where_value反推header
    ############2.基于汉字的columns提供三个候选，基于数字的column提供三个候选，并且再提供三个候补，用于特殊情况的备选
    re_list = re_val()
    [bb_more_list,bb_less_list,bb_more_spe_list,bb_less_spe_list] = [re_list[i] for i in range(0,4)]
    [hhh_more_list1, hhh_less_list1,hhh_more_spe_list1,hhh_less_spe_list1] = [re_list[i] for i in range(4,8)]
    [hhh_more_list2, hhh_less_list2,hhh_more_spe_list2,hhh_less_spe_list2] = [re_list[i] for i in range(8,12)]
    with open(f, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            print(idx)
            sql = json.loads(line.strip())
            table_id = sql['table_id']
            question = sql['question']
            columns,header = column_match(question,table_id)
            out_dict['column'] = columns
            out_dict['header'] = header
            out_dict['number'] = [bb_more_list[idx],bb_less_list[idx],bb_more_spe_list[idx],bb_less_spe_list[idx]]   
            out_dict['num_column'] = [hhh_more_list1[idx], hhh_less_list1[idx],hhh_more_spe_list1[idx],hhh_less_spe_list1[idx]]
            out_dict['num_column_waiting'] = [hhh_more_list2[idx], hhh_less_list2[idx],hhh_more_spe_list2[idx],hhh_less_spe_list2[idx]]
            out_dict['conds'] = get_input_dict(idx)['conds']
            out_dict_list.append(out_dict)
    inf.close()
    return out_dict_list

def second_step(out_dict_list):
    fw = open('../data/pre_val.json','r+',encoding='utf-8')
    for idx, line in enumerate(out_dict_list):
        print(idx)
        ##############################1.首先拿到最好的sel部分和where_num部分进行组合
        sql = line.strip()  
        column = sql['column']
        header = sql['header']
        number = sql['number']
        input_dict = get_input_dict(idx)  ###############包含最好的select部分和where_num部分
        conds = input_dict['conds']
        sel = input_dict['sel']
        num_column = sql['num_column']
        num_column_waiting = sql['num_column_waiting']   ############候选，当和sel选择的列冲突时启用该列表
        ########################2.根据where_num确定first_step中到底要取几个值，组合成完整的结构化的mysql语句
        new_conds = []
        tem_conds = []
        cond_num = 0   #已经写入了几个cond
        for i in range(0,4):
            if len(number[i])>0:
                for j ,sub_num in enumerate(number[i]):
                    sub_cond = []
                    if i == 0 or i == 2:
                        cond_op = 0
                    else:
                        cond_op = 1
                    if len(num_column[i]) >j:   #############应对 都 这种情况，预测出来的num_column的数目要多于number
                        if num_column[i][j] not in sel and num_column[i][j] not in tem_conds:
                            sub_cond.append(num_column[i][j])
                        else:
                            sub_cond.append(num_column_waiting[i][j])
                    else:
                        if num_column[i][j-1] not in sel and num_column[i][j-1] not in tem_conds:
                            sub_cond.append(num_column[i][j-1])
                        else:
                            sub_cond.append(num_column_waiting[i][j-1])
                    tem_conds.extend(sub_cond)
                    sub_cond.append(cond_op)
                    sub_cond.append(sub_num)
                    new_conds.append(sub_cond)
                    cond_num += 1
                    if cond_num >= len(conds):
                        break
                    
                if cond_num >= len(conds):
                    break
        avi_len = len(conds) - cond_num
        j = 0
        for k in range(0,avi_len):
            sub_cond = []
            if j < len(header) and header[j] not in sel:
                sub_cond.append(header[j])
                sub_cond.append(2)
                sub_cond.append(column[j])
                new_conds.append(sub_cond)
                j = j+1
            else:
                if j+1 < len(header):
                    sub_cond.append(header[j+1])
                    sub_cond.append(2)
                    sub_cond.append(column[j+1])
                    new_conds.append(sub_cond)

                j = j + 2
            if j >= avi_len:
                break
        ############3.根据MySQL语句本身的规则做一些assertion提高准确率
        ###  去除00000000000000
        real_new_conds = []
        if len(new_conds) > 1:
            for i,sub_cond in enumerate(new_conds):
                if '000000000000' in sub_cond:
                    real_new_conds = conds
                    break
                    #real_new_conds.append(sub_cond)
                else:
                    real_new_conds.append(sub_cond)
        elif len(new_conds) == 1:
            if '000000000000' in new_conds[0] :
                real_new_conds = conds
            else:
                real_new_conds = new_conds
        else:
            real_new_conds = conds
                
        input_dict['conds'] = real_new_conds
        ####################特殊规则
        if len(real_new_conds) > 1 and real_new_conds[0][2]==real_new_conds[1][2] and is_number(real_new_conds[0][2]) == False:                
            input_dict['conds'] = [real_new_conds[0]]
            print(conds)
        
        if len(real_new_conds) == 2 and real_new_conds[0]==real_new_conds[1]:
            input_dict['conds'] = conds
        if len(real_new_conds) == 1:
            input_dict['cond_conn_op'] = 0
        if len(real_new_conds) >1 and input_dict['cond_conn_op'] == 0:
            input_dict['cond_conn_op'] = 1
        if len(real_new_conds) == 2 and real_new_conds[0][0]==real_new_conds[1][0] :
            input_dict['cond_conn_op'] = 2
        
        a = json.dumps(input_dict, ensure_ascii=False)+'\n'
        fw.write(a)

if __name__ == '__main__':             
    ############1.用于预测where_column和where_value部分，并根据where_value反推header
    ############2.基于汉字的columns提供三个候选，基于数字的column提供三个候选，并且再提供三个候补，用于特殊情况的备选
    out_dict_list = first_step()
    ############1.首先拿到最好的sel部分和where_num部分进行组合
    ############2.根据where_num确定first_step中到底要取几个值，组合成完整的结构化的mysql语句
    ############3.根据MySQL语句本身的规则做一些assertion提高准确率
    second_step(out_dict_list)   
    
