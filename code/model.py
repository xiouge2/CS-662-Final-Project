#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import pickle
import sys
from measure import Measure
from utils import read_data

class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, num_entity, num_relation, embedding_dim, 
                 lstm_units, hidden_dim, num_classes, lstm_layers,
                 bidirectional, dropout, batch_size):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.entity_embedding = nn.Embedding(num_entity+1, embedding_dim, padding_idx=num_entity)
        self.relation_embedding = nn.Embedding(num_relation+1, embedding_dim, padding_idx=num_relation)
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_units,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units


    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units).to(device)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units).to(device)))
        return h, c

    def forward(self, h_t_id, path_r_id, device):
        batch_size = h_t_id.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        head_batch = self.entity_embedding(h_t_id[:, 0])
        tail_batch = self.entity_embedding(h_t_id[:, 1])
        path_relation_batch = self.relation_embedding(path_r_id)
        head_batch = head_batch.view(head_batch.shape[0], 1, head_batch.shape[1])
        # print(head_batch.shape)
        tail_batch = tail_batch.view(tail_batch.shape[0], 1, tail_batch.shape[1])
        # print(tail_batch.shape)
        path_length = 2 + path_relation_batch.shape[1]
        padding = torch.zeros(head_batch.shape[0], 7-path_length, head_batch.shape[2]).to(device)
        # print(path_relation_batch.shape)
        # print(padding.shape)
        padded_embed_input = torch.cat((head_batch, path_relation_batch, tail_batch, padding), dim=1)
        # print(padded_embed_input.shape)
        # print(embedding_input.shape)
        text_lengths = torch.ones(batch_size) * (path_length)
        packed_embedded = pack_padded_sequence(padded_embed_input, text_lengths, batch_first=True)
        # print(packed_embedded[0].shape)
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

# def train(model, optimizer, loss_function, train_loader, train_size):

def evaluate(data_path, mode, device):
    measure = Measure()
    for length in [2,3,4,5]:
        save_file_name = os.path.join(data_path, mode+'_length_'+str(length)+'_data.pt')
        tensor_Dataset = torch.load(save_file_name)
        data_loader = DataLoader(tensor_Dataset, batch_size=len(tensor_Dataset))
        for i, (h_t_id, path_r_id, labels) in enumerate(data_loader):
            h_t_id, path_r_id = h_t_id.to(device), path_r_id.to(device)
            outputs = model.forward(h_t_id, path_r_id, device)
            score_ranking = torch.argsort(outputs.data, dim=1, descending=True)
            labels = labels.to(device)
            rank = (score_ranking == labels.view(-1, 1)).nonzero()[:,1] + 1.0
            measure.update(rank)
    measure.normalize()
    print(mode+' performance')
    measure.print_()
    
  
if __name__ == '__main__':
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    print(dev)
    device = torch.device(dev)  
    parser = argparse.ArgumentParser(description="Get input data")
    parser.add_argument("-i",  help="inputfile: the name/path of the test file that has to be read one text per line")
    args = parser.parse_args()
    dataset = args.i
    batch_size = 100
    num_of_epoch = 10
    cwd = os.getcwd()
    root = os.path.dirname(cwd)
    data_path = os.path.join(root, 'data', dataset)
    with open(os.path.join(data_path, 'idx_maps'), 'rb') as file:
        idx_maps = pickle.load(file)
    num_entity = len(idx_maps['entity2id'])
    num_relation = len(idx_maps['relation2id'])
    embedding_dim = 300
    lstm_units = 300
    hidden_dim = 300
    num_classes = num_relation
    lstm_layers = 1
    bidirectional = True
    dropout = 0.1
    text_lengths = 7
    model = LSTM(num_entity, num_relation, embedding_dim, 
                 lstm_units, hidden_dim, num_classes, lstm_layers,
                 bidirectional, dropout, batch_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    mode = 'dev'
    # for mode in ['dev']:#, 'dev', 'test']:
    for epoch in range(num_of_epoch):
        epoch_loss = 0.0
        epoch_acc = 0.0
        train_size = 0
        for length in [2,3,4,5]:
            save_file_name = os.path.join(data_path, mode+'_length_'+str(length)+'_data.pt')
            tensor_Dataset = torch.load(save_file_name)
            # print(tensor_Dataset)
            data_loader = DataLoader(tensor_Dataset, batch_size=batch_size, shuffle=True)
            train_size += len(data_loader)
            # sys.exit()
            for i, (h_t_id, path_r_id, labels) in enumerate(data_loader):
                optimizer.zero_grad()
                h_t_id, path_r_id = h_t_id.to(device), path_r_id.to(device)
                outputs = model.forward(h_t_id, path_r_id, device)
                labels = labels.to(device)
                # outputs = model.forward(h_t_id, path_r_id, text_lengths)
                loss = loss_function(outputs, labels).to(device)
                _, preds = torch.max(outputs.data, 1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # epoch_acc += torch.sum(preds == labels).item()
        epoch_loss /= train_size
        print('epoch_'+str(epoch)+': '+str(epoch_loss))
            # print(epoch_acc)
        # epoch_acc /= train_size
        evaluate(data_path, 'dev', device)
    evaluate(data_path, 'test', device)
    model_file = os.path.join(cwd, dataset+'_model.pt')
    torch.save(model.state_dict(), model_file)
            # return epoch_loss, epoch_acc 
    # train_tensorDataset, dev_tensorDataset, test_tensorDataset = read_data(args)
    # for epoch in range(num_of_epoch):
    #     for length, tensor_Dataset in train_tensorDataset.items():
    #         data_loader = DataLoader(tensor_Dataset, batch_size=batch_size, shuffle=True)
    #         # model.train()
    #         epoch_loss = 0.0
    #         epoch_acc = 0.0
    #         for i, (h_t_id, path_r_id, labels) in enumerate(data_loader):
    #             print(h_t_id, path_r_id, labels)
    #             sys.exit()
                # print(features.shape)
            #     optimizer.zero_grad()
            #     outputs = model.forward(word_idx, pos_idx, arc_idx)
            #     loss = loss_function(outputs, labels)
            #     _, preds = torch.max(outputs.data, 1)
            #     loss.backward()
            #     optimizer.step()
            #     epoch_loss += loss.item()
            #     epoch_acc += torch.sum(preds == labels).item()
            # epoch_loss /= train_size
            # epoch_acc /= train_size
            # return epoch_loss, epoch_acc 