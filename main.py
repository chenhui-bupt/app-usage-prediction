# -*- coding: utf-8 -*-
import itertools
import random
import numpy as np
import pandas as pd

from base.data_loader import load_data, gen_graph
from evaluation.evaluate import eval_all


def main():
    print("Start the App usage prediction task!")
    print("loading data ...")
    data, all_data = load_data()
    print("data is loaded!")
    print("constructing the User-App networks ...")
    G1 = gen_graph(data[0], graph_id=1)
    G2 = gen_graph(data[1], graph_id=2)
    print("User-App networks have constructed sucessfully!")
    print("calculating the neighbor based similarity of all User-App networks ...")
    nb_data1 = G1.neighbor_based_sim(all_data[0])
    nb_data2 = G2.neighbor_based_sim(all_data[1])
    nb_data1.to_csv('./output/node/nodesim1.csv', encoding='gbk', index=False)
    nb_data2.to_csv('./output/node/nodesim2.csv', encoding='gbk', index=False)

    G1.node2id()
    G2.node2id()

    print("calculating the local path similarity of all User-App networks ...")
    lp_data1 = G1.local_path(all_data[0])
    lp_data2 = G2.local_path(all_data[1])
    lp_data1.to_csv('./output/path/lp1.csv', encoding='gbk', index=False)
    lp_data2.to_csv('./output/path/lp2.csv', encoding='gbk', index=False)

    print("calculating the local shortest path similarity of all User-App networks ...")
    lsp_data1 = G1.local_shortest_path(all_data[0], feature_name='LSP')
    lsp_data2 = G2.local_shortest_path(all_data[1], feature_name='LSP')
    lsp_data1.to_csv('./output/path/lsp1.csv', encoding='gbk', index=False)
    lsp_data2.to_csv('./output/path/lsp2.csv', encoding='gbk', index=False)

    print("calculating the random_walk_with_restart similarity of all User-App networks ...")
    rwr_data1 = G1.random_walk_with_restart(all_data[0], c=0.85, feature_name='RWR')
    rwr_data2 = G2.random_walk_with_restart(all_data[1], c=0.85, feature_name='RWR')
    rwr_data1.to_csv('./output/randomwalk/rwr1.csv', encoding='gbk', index=False)
    rwr_data2.to_csv('./output/randomwalk/rwr2.csv', encoding='gbk', index=False)

    print("calculating the random_walk_with_resource_redistribution similarity of all User-App networks ...")
    rwrr_data1 = G1.random_walk_with_resource_redistribution(all_data[0], c=0.85, feature_name='RWRR')
    rwrr_data2 = G2.random_walk_with_resource_redistribution(all_data[1], c=0.85, feature_name='RWRR')
    rwrr_data1.to_csv('./output/randomwalk/rwrr1.csv', encoding='gbk', index=False)
    rwrr_data2.to_csv('./output/randomwalk/rwrr2.csv', encoding='gbk', index=False)

    # evaluate the app usage prediction using different models
    print("predicting the App usage of all users using different models and evaluate it ...")
    eval_all()
    print("Finished!")


if __name__ == '__main__':
    main()





