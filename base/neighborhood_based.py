# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
import math


def dict_of_neighbors(G):
    nodes=G.nodes()
    neighbors={} # its value's type is list
    for  node in nodes:
        neighbors[node]=G.neighbors(node)
    neighbors_of_neighbors={} # its value's type is set
    for node in nodes:
        tmp=set(neighbors[neighbors[node][0]])# init neighbors of neighbors[0]
        for neig in neighbors[node]:
            tmp=tmp&set(neighbors[neig])
        neighbors_of_neighbors[node]=tmp
    return neighbors,neighbors_of_neighbors


def get_neighbors(G, num):
    import os,pickle
    path='./cache/neighbors%s.pkl'%num
    if os.path.exists(path):
        f=open(path,'rb')
        neighbors,neighbors_of_neighbors=pickle.load(f)
        f.close()
    else:
        neighbors,neighbors_of_neighbors=dict_of_neighbors(G)
        f=open(path,'wb')
        pickle.dump((neighbors,neighbors_of_neighbors),f)#将两个邻居字典作为一个元组存入序列化文件
        f.close()
    return neighbors,neighbors_of_neighbors


def nodesim(G,x,y,neighbors,neighbors_of_neighbors):
    try:
        sd=nx.shortest_path_length(G,x,y) # SD
    except:
        sd=99
    rx = set(neighbors[x]) # set(x)
    rry = neighbors_of_neighbors[y] # set(y)
    cn=rx&rry
    tn=rx|rry
    cn_len=len(cn)
    jc=cn_len*1.0/len(tn) if len(tn)>0 else 0 # JC
    aa=0.0
    ra=0.0
    for z in cn:
        if len(neighbors[z])==1:
            ra += 1.0/len(neighbors[z]) # Resource Allocation（RA)
            continue
        aa += 1.0/math.log(len(neighbors[z])) # Adamic-Adar(AA)
        ra += 1.0/len(neighbors[z]) # Resource Allocation（RA)
    pa=len(rx)*len(rry)  # Preferential Attachment(PA)
    cs=cn_len/math.sqrt(pa) if pa>0 else 0 # CS,Salton
    lhn=cn_len*1.0/pa if pa>0 else 0 # Leicht-Holme-Newman Index
    hpi=cn_len*1.0/min(len(rx),len(rry)) if pa>0 else 0# Hub Promoted Index(HPI)
    hdi=cn_len*1.0/max(len(rx),len(rry)) if max(len(rx),len(rry))>0 else 0# Hub Depressed Index(HDI)
    si=cn_len*2.0/(len(rx)+len(rry)) if (len(rx)+len(rry))>0 else 0 #Sorence index
    cp=cn_len*1.0/len(rry) if len(rry)>0 else 0# Condition probability(CP)
    return sd,cn_len,jc,aa,ra,pa,cs,lhn,hpi,hdi,si,cp


def add_sim_to_edges(G,all_data,neighbors,neighbors_of_neighbors):
    array_data=np.array(all_data)
    sm=[nodesim(G,x[0],x[1],neighbors,neighbors_of_neighbors)+nodesim(G,x[1],x[0],neighbors,neighbors_of_neighbors) for x in array_data]
    cols=['SD','CN','JC','AA','RA','PA','CS','LHN','HP','HD','SI','CP','SD_2','CN_2','JC_2','AA_2','RA_2','PA_2','CS_2','LHN_2','HP_2','HD_2','SI_2','CP_2']
    sm_data=pd.DataFrame(sm,columns=cols)
    sm_data=pd.concat([all_data,sm_data],1)
    return sm_data