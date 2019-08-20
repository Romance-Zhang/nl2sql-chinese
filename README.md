## Introduction
这个项目是追一科技的nl2sql比赛项目，任务是输入一个自然语言问句，输出一个相对应的mysql语句。
###整个任务分为八个子任务：
####1.SELECT NUM
####2.SELECT COLUMN
####3.SELECT AGG
####4.WHERE NUM
####5.WHERE COLUMN
####6.WHERE VALUE
####7.WHERE OP
####8.WHERE RELATION

整个代码可以分为两个部分：一部分是基于深度学习的模型,使用的是bert+sqlnet,参考sqlove和追一科技提供的baseline，该模型用于给出除了WHERE COLUMN
和WHERE VALUE之外的其他子任务的预测；另一部分是基于规则的模型，用于预测WHERE COLUMN和WHERE VALUE部分，并基于规则对其他某些子任务作出部分修正，
主要代码文件在rule_base.py

## Dependencies
 - babel
 - matplotlib
 - defusedxml
 - tqdm
 - Python 3.7
 - torch 1.0.1
 - records 0.5.3

## 训练

data目录下是数据文件，结构如下图所示：
```
├── data_nl2sql
│ ├── train
│ │ ├── train.db
│ │ ├── train.json
│ │ ├── train.tables.json
│ ├── val
│ │ ├── val.db
│ │ ├── val.json
│ │ ├── val.tables.json
│ ├── test
│ │ ├── test.db
│ │ ├── test.json
│ │ ├── test.tables.json
│ ├── char_embedding.json
```
首先需要下载中文bert文件放在code/目录下

启动命令：python code/train.py --ca --gpu --fine_tune
训练好的模型会保存在saved_model目录下

## 预测
### 第一部分
首先得到val集的深度学习模型预测结果
启动命令：python code/test.py --ca --gpu --output_dir data/best_val.json
预测结果会保存在data路径下best_val文件中

### 第二部分
对best_val中的结果做第二步处理，基于规则预测WHERE COLUMN和WHERE VALUE，并对WHERE OP和WHERE RELATION做相应修正
启动命令：python code/rule_base_val.py 
会写入中间文件lalala_val.json以及生成最终的预测结果文件pre_val.json

### 第三部分
得出验证集上的准确率预测结果
启动命令：python code/test.py --ca --gpu --mode_type 2 --output_dir xxx.json

### 测试集的预测只需要针对以上做相应修改即可

## 实验结果
线上测试集的平均准确率在72%左右，验证集的平均准确率会稍高一点

## to_do
基于规则的预测中需要生成中间文件，有一些繁琐，接下来会改进中间过程，提高简便性。
rule_base.py的代码精简，现在可读性比较差
