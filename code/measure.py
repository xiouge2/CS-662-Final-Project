#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Measure:
    def __init__(self):
        self.hit1 = 0.0
        self.hit3 = 0.0
        self.hit10 = 0.0
        self.mrr = 0.0
        self.mr = 0.0
        self.num_facts = 0.0

    def update(self, rank):
        self.hit1 += (rank==1).sum().item()
        self.hit3 += (rank<=3).sum().item()
        self.hit10 += (rank<=10).sum().item()
        self.mr += rank.sum().item()
        self.mrr += (1.0 / rank).sum().item()
        self.num_facts += rank.shape[0]

    def normalize(self):
        if self.hit1 > 0.0:
            self.hit1 /= self.num_facts
        if self.hit3 > 0.0:
            self.hit3 /= self.num_facts
        if self.hit10 > 0.0:
            self.hit10 /= self.num_facts
        self.mr /= self.num_facts
        self.mrr /= self.num_facts

    def print_(self):
        print("\tHit@1 =", self.hit1)
        print("\tHit@3 =", self.hit3)
        print("\tHit@10 =", self.hit10)
        print("\tMR =", self.mr)
        print("\tMRR =", self.mrr)
        print("")
