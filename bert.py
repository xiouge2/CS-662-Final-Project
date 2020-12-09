import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
# from tqdm import tqdm, trange
from tqdm.notebook import tqdm, tnrange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

 class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

 class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


 class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                triple_label = line[3]
                if triple_label == "1":
                    label = "1"
                else:
                    label = "0"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                        if tmp_triple_str not in lines_str_set:
                            break                    
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


if __name__ == '__main__':
    task_name = "kg"
    processors = {
        "kg": KGProcessor,
    }
    data_dir = '/content/drive/My Drive/WN11path'
    bert_model = "bert-base-uncased"
    do_lower_case = True
    do_train = True
    processor = processors[task_name]()
    num_labels = 22
    train_examples = None
    num_train_optimization_steps = 0
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
    with open('/content/drive/My Drive/WN11/train_examples', 'rb') as handle:
        train_examples = pickle.load(handle)
    train_batch_size = 32
    gradient_accumulation_steps = 1
    num_train_epochs = 1.0
    local_rank = -1
    with open('/content/drive/My Drive/WN11path/train_features', 'rb') as handle:
        train_features = pickle.load(handle)
    num_train_optimization_steps = int(len(train_features) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    cache_dir = '/content/drive/My Drive/WN11path'
    model = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=cache_dir, num_labels=num_labels)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    learning_rate = 5e-5
    warmup_proportion = 0.1
    optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion, t_total=num_train_optimization_steps)
    max_seq_length = 50
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    model.to(device)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    for _ in tnrange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        #print(logits, logits.shape)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            # if args.fp16:
            #     # modify learning rate with special warm up BERT uses
            #     # if args.fp16 is False, BertAdam is used that handles this automatically
            #     lr_this_step = learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps, warmup_proportion)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    tokenizer.save_vocabulary(output_dir)
    # eval_examples = processor.get_dev_examples(data_dir)
# eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
    with open('/content/drive/My Drive/WN11path/dev_features', 'rb') as handle:
        eval_features = pickle.load(handle)
    output_dir = '/content/drive/My Drive/WN11path/'
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
    model.to(device)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        print(label_ids.view(-1))
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(task_name, preds, all_label_ids.numpy())
    loss = tr_loss/nb_tr_steps if do_train else None

    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    result['loss'] = loss

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
    ent2text = {}
    with open("/content/drive/My Drive/WN11path/entity2text.txt", 'r') as f:
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                end = temp[1]#.find(',')
                ent2text[temp[0]] = temp[1]

    rel2text = {}
    with open("/content/drive/My Drive/WN11path/relation2text.txt", 'r') as f:
        rel_lines = f.readlines()
        for line in rel_lines:
            temp = line.strip().split('\t')
            rel2text[temp[0]] = temp[1]
    rel2index = dict(zip(rel2text,list(range(len(rel2text)))))
    entity_pairs = set()
    entity_pairs_relation_map = {}
    with open('/content/drive/My Drive/WN11path/train.txt') as triple_file:
        for line in triple_file.readlines():
            h, r, t = line.split()
            entity_pairs.add((h,t))
            if (h,t) in entity_pairs_relation_map:
                entity_pairs_relation_map[(h,t)].add(r)
            else:
                entity_pairs_relation_map[(h,t)] = set([r])
            if (t,h) in entity_pairs_relation_map:
                entity_pairs_relation_map[(t,h)].add('**'+r)
            else:
                entity_pairs_relation_map[(t,h)] = set(['**'+r])
    closed_paths = []
    closed_paths_format = []
    with open('/content/drive/My Drive/WN11path/test_path') as path_file:
        for line in path_file.readlines():
            path = line.split()
            h,t = path[0],path[-1]
            if (h,t) in entity_pairs_relation_map:
                path_length = len(path[1].split(','))
                if path_length>1:
                    for relation in entity_pairs_relation_map[(h,t)]:
                        closed_paths.append([path[1].split(','), relation])
                        closed_paths_format.append([[ent2text[h]]+list(map(rel2text.get, path[1].split(',')))+[ent2text[t]], rel2index[relation]])
    features = []
for example in closed_paths_format:
    # input_ids: convert from token to ids
    # segment_ids: use 0 1 to represent different entity and relation the path
    # input_mask: use 1 0 to represent real tokens and padding
    max_seq_length = 50
    tokens = ["[CLS]"]
    segment_flag = 0
    segment_ids = [segment_flag]
    path, label = example
    for name in path:
        # convert to tokens
        tokenized_name = tokenizer.tokenize(name)
        tokens+=tokenized_name+["[SEP]"]
        # set segment_ids
        segment_ids+=[segment_flag]*(len(tokenized_name)+1)
        if segment_flag == 0:
            segment_flag = 1
        else:
            segment_flag = 0
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids
    input_mask = [1]*len(tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    segment_ids += padding
    input_mask += padding
    features.append(
                InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=label))

    test_batch_size = 8
    test_features = features
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
    model.to(device)

    model.eval()
    test_loss = 0
    nb_test_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create test_ loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_test_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        print(label_ids.view(-1))
        
        test_loss += tmp_test_loss.mean().item()
        nb_test_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    test_loss = test_loss / nb_test_steps
    preds = preds[0]

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(task_name, preds, all_label_ids.numpy())
    loss = tr_loss/nb_tr_steps if do_train else None

    result['test_loss'] = test_loss
    result['global_step'] = global_step
    result['loss'] = loss

    output_test_file = os.path.join(output_dir, "test_results.txt")
    with open(output_test_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))