# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:08:21 2019

@author: TsungYuan
"""
import networkx as nx 
import matplotlib.pyplot as plt 

import numpy as np
np.set_printoptions(threshold=np.inf)

#------------ read_data ------------
with open("C://Users//TsungYuan//Desktop//datamining//DM_Project3_N96071172//hw3dataset//graph_1.txt", 'r') as log_fp:
    logs = [ log.strip() for log in log_fp.readlines() ]
'''
logs_tuple = []
for i in range(len(logs)-1):
    logs_tuple.append(tuple([int(logs[i].split("       ")[2]),int(logs[i+1].split("       ")[2])]))
    logs_tuple.append(tuple([int(logs[i+1].split("       ")[2]),int(logs[i].split("       ")[2])]))
'''
logs_tuple = [ tuple(log.split(",")) for log in logs ]

#G = nx.Graph() 
DG = nx.DiGraph()
DG.add_edges_from(logs_tuple) 

#------------ draw_graph ------------
plt.figure(figsize =(10, 10)) 
nx.draw_networkx(DG, with_labels = True) 

#------------ hits ------------

hubs, authorities = nx.hits(DG, max_iter = 50, normalized = True)
print("Hub Scores: ", hubs) 
print("Authority Scores: ", authorities)

#------------ pagerank ------------
'''
pr = nx.pagerank(DG, alpha=0.85)
print("Page Rank: ",pr)
'''

#------------ simrank ------------
'''
def get_ads_num(query):
    q_i = queries.index(query)
    return graph[q_i]

def get_queries_num(ad):
    a_j = ads.index(ad)
    return graph.transpose()[a_j]

def get_ads(query):
    series = get_ads_num(query).tolist()[0]
    return [ ads[x] for x in range(len(series)) if series[x] > 0 ]

def get_queries(ad):
    series = get_queries_num(ad).tolist()[0]
    return [ queries[x] for x in range(len(series)) if series[x] > 0 ]


def query_simrank(q1, q2, C):
    if q1 == q2 : return 1
    prefix = C / (get_ads_num(q1).sum() * get_ads_num(q2).sum())
    postfix = 0
    for ad_i in get_ads(q1):
        for ad_j in get_ads(q2):
            i = ads.index(ad_i)
            j = ads.index(ad_j)
            postfix += ad_sim[i, j]
    return prefix * postfix
    

def ad_simrank(a1, a2, C):
    if a1 == a2 : return 1
    prefix = C / (get_queries_num(a1).sum() * get_queries_num(a2).sum())
    postfix = 0
    for query_i in get_queries(a1):
        for query_j in get_queries(a2):
            i = queries.index(query_i)
            j = queries.index(query_j)
            postfix += query_sim[i,j]
    return prefix * postfix


def simrank(C=0.8, times=1):
    global query_sim, ad_sim

    for run in range(times):
        # queries simrank
        new_query_sim = matrix(numpy.identity(len(queries)))
        for qi in queries:
            for qj in queries:
                i = queries.index(qi)
                j = queries.index(qj)
                new_query_sim[i,j] = query_simrank(qi, qj, C)

        # ads simrank
        new_ad_sim = matrix(numpy.identity(len(ads)))
        for ai in ads:
            for aj in ads:
                i = ads.index(ai)
                j = ads.index(aj)
                new_ad_sim[i,j] = ad_simrank(ai, aj, C)

        query_sim = new_query_sim
        ad_sim = new_ad_sim


import numpy
from numpy import matrix
queries = list(set([ log[0] for log in logs_tuple ]))
ads = list(set([ log[1] for log in logs_tuple ]))
graph = numpy.matrix(numpy.zeros([len(queries), len(ads)]))

for log in logs_tuple:
    query = log[0]
    ad = log[1]
    q_i = queries.index(query)
    a_j = ads.index(ad)
    graph[q_i, a_j] += 1
#print (graph)

query_sim = matrix(numpy.identity(len(queries)))
ad_sim = matrix(numpy.identity(len(ads)))
simrank()
print("query_sim: ",query_sim)
print("ad_sim: ",ad_sim)
'''
