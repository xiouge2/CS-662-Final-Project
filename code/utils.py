#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os.path.dirname as dirname
import os
import argparse
import pickle
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_id_maps(entity_file, relation_file):
    with open(entity_file) as file:
        entity_array = file.read().split('\n')
    entity2id = dict(zip(entity_array, list(range(len(entity_array)))))
    with open(relation_file) as file:
        relation_array = file.read().split('\n')
    relation2id = dict(zip(relation_array, list(range(len(relation_array)))))
    return entity2id, relation2id

def get_closed_path_rules_stats(triple_file_names, path_file_name):
    entity_pairs = set()
    entity_pairs_relation_map = {}
    for (triple_file_name, mode) in triple_file_names:
        with open(triple_file_name) as triple_file:
            for line in triple_file.readlines():
                if mode == 'train':
                    h, r, t = line.split()
                else:
                    h, r, t, l = line.split()
                    if l == '-1':
                        continue
                entity_pairs.add((h,t))
                if (h,t) in entity_pairs_relation_map:
                    entity_pairs_relation_map[(h,t)].add(r)
                else:
                    entity_pairs_relation_map[(h,t)] = set([r])
    closed_paths_rules = []
    paths_by_length = {}
    with open(path_file_name) as path_file:
        for line in path_file.readlines():
            path = line.split()
            h,t = path[0],path[2]
            if (h,t) in entity_pairs_relation_map:
                path_length = len(path[1].split(','))
                if path_length not in paths_by_length:
                    paths_by_length[path_length] = []
                if path_length>1:
                    for relation in entity_pairs_relation_map[(h,t)]:
                        closed_paths_rules.append([[h,t], path[1].split(','), relation])
                        paths_by_length[path_length].append([[h,t], path[1].split(','), relation])
    return closed_paths_rules, paths_by_length

def write_readable_paths(closed_paths_rules, mode, data_path):
    # write in readable format
    file_path = os.path.join(data_path, mode+'.converted')
    with open(file_path, 'w') as output:
        for sample in closed_paths_rules:
            output.write(str(sample)+'\n')
    
def get_tensor_dataset(paths_by_length, entity2id, relation2id, mode, data_path):
    tensorDataset_by_length = {}
    for length, rules in paths_by_length.items():
        all_h_t = []
        all_path_r = []
        all_label = []
        for [head_and_tail, path, predict] in rules:
            # print(head_and_tail, path, predict)
            h_t_id = list(map(entity2id.get, head_and_tail))
            path_r_id = list(map(relation2id.get, path))
            predict_r_id = relation2id[predict]
            all_h_t.append(h_t_id)
            all_path_r.append(path_r_id)
            all_label.append(predict_r_id)
        try:    
            dataset = TensorDataset(torch.LongTensor(all_h_t), 
                                    torch.LongTensor(all_path_r),
                                    torch.LongTensor(all_label)
                                    )
        except:
            print(rules[1])
        save_file_name = os.path.join(data_path, mode+'_length_'+str(length)+'_data.pt')
        torch.save(dataset, save_file_name)
        tensorDataset_by_length[length] = dataset
    return tensorDataset_by_length
    
def read_data(args):
    dataset = args.i
    cwd = os.getcwd()
    root = os.path.dirname(cwd)
    data_path = os.path.join(root, 'data', dataset)
    entity_file = os.path.join(data_path, 'entities.txt')
    relation_file = os.path.join(data_path, 'relations.txt')
    train_triple_file = os.path.join(data_path, 'train.txt')
    dev_triple_file = os.path.join(data_path, 'dev.txt')
    test_triple_file = os.path.join(data_path, 'test.txt')
    train_path_file = os.path.join(data_path, 'train')
    dev_path_file = os.path.join(data_path, 'dev')
    test_path_file = os.path.join(data_path, 'test')
    entity2id, relation2id = get_id_maps(entity_file, relation_file)
    idx_maps = {'entity2id': entity2id,
                'relation2id': relation2id}
    with open(os.path.join(data_path, 'idx_maps'), "wb") as file:
         pickle.dump(idx_maps, file)
    triple_file_names = [(train_triple_file, 'train'), 
                          (dev_triple_file, 'dev'), 
                          (test_triple_file, 'test')]
    train_data = get_closed_path_rules_stats(triple_file_names, train_path_file)
    train_closed_paths_rules, train_paths_by_length = train_data
    dev_data = get_closed_path_rules_stats(triple_file_names, dev_path_file)
    dev_closed_paths_rules, dev_paths_by_length = dev_data
    test_data = get_closed_path_rules_stats(triple_file_names, test_path_file)
    test_closed_paths_rules, test_paths_by_length = test_data    
    total = 0
    for key, value in train_paths_by_length.items():
        print(key, len(value))
        total += len(value)
    print('total', total)
    print('\n')
    total = 0
    for key, value in dev_paths_by_length.items():
        print(key, len(value))
        total += len(value)
    print('total', total)
    print('\n')
    total = 0
    for key, value in test_paths_by_length.items():
        print(key, len(value))
        total += len(value)
    print('total', total)
    write_readable_paths(train_closed_paths_rules, 'train', data_path)
    write_readable_paths(dev_closed_paths_rules, 'dev', data_path)
    write_readable_paths(test_closed_paths_rules, 'test', data_path)
    
    train_tensorDataset = get_tensor_dataset(train_paths_by_length, entity2id, relation2id, 'train', data_path)
    dev_tensorDataset = get_tensor_dataset(dev_paths_by_length, entity2id, relation2id, 'dev', data_path)
    test_tensorDataset = get_tensor_dataset(test_paths_by_length, entity2id, relation2id, 'test', data_path)
    return train_tensorDataset, dev_tensorDataset, test_tensorDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get input data")
    parser.add_argument("-i",  help="inputfile: the name/path of the test file that has to be read one text per line")
    args = parser.parse_args()
    read_data(args)
