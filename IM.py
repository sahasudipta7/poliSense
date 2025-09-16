import matplotlib.pyplot as plt;
from random import uniform,seed;
import numpy as np;
import time;
import networkx as nx;
import heapq;
import random;

#spread process(LT)

def LT(DG,I,th=0.1,mc=1000):
    spread=[];
    for i in range(mc):
        newInfected, currInfectedSet=I[:],I[:];
        while newInfected:
            #print(newInfected);
            nowInfected=[];
            for node in DG.nodes():
                if node in currInfectedSet:
                    continue;
                np.random.seed(i);
                th=np.random.uniform(0,0.1);#threshhold
                #print("th",th);
                thNodes=(th)*DG.number_of_nodes();
                cnt=0;
                for adjNode in DG.neighbors(node):
                    if adjNode in currInfectedSet:
                        cnt+=1;
                if cnt>thNodes:
                    nowInfected.append(node);
            newInfected=nowInfected;
            #print(nowInfected);
            currInfectedSet+=(newInfected);
        spread.append(len(currInfectedSet));
        #print(spread);
    return np.mean(spread);
# spread process(IC)
def IC(DG,S,p=0.5,mc=1000):
    spread=[];
    for i in range(mc):
        newActive, currSeedSet = S[:], S[:];
        while newActive:
            newInfluenced=[];
            for node in newActive:
                np.random.seed(i);
                #success = np.random.uniform(0, 1, len(DG.neighbors(node,mode="out")));#igraph version
                #success = np.random.uniform(0, 1, len(DG.neighbors(node)));
                success = np.random.uniform(0, 1, len(list(DG.successors(node))))#0.1 instead of 1
                #newInfluenced+=list(np.extract(success,DG.neighbors(node,mode="out")));
                newInfluenced += list(np.extract(success,list(DG.successors(node))));
            newActive=list(set(newInfluenced)-set(currSeedSet));
            currSeedSet+=newActive;
            #print(currSeedSet);
        spread.append(len(currSeedSet));
    return np.mean(spread);

# greedy algotithm seed selection
def greedy(DG,k,p=0.1,mc=1000):
    S,spread,timeLapse,startTime=[],[],[],time.time();
    for i in range(k):
        bestSpread=0;
        node=None;
        for j in set(DG.nodes())-set(S):
            #print("j: ",j);
            newSpread=IC(DG,S+[j],p,mc);
            spreadSeedSet=IC(DG,S,p,mc);
            marginalSpread=newSpread-spreadSeedSet;
            #print("j: ",j,"newSpread: ",newSpread,"bestSpread: ",bestSpread,"seedset: ",S+[j]);
            if marginalSpread>bestSpread:
                bestSpread,node=newSpread,j;

        #print("node: ",node);
        if node!=None:
            S.append(node);
        spread.append(bestSpread);
        timeLapse.append(time.time()-startTime);
    return (S,spread,timeLapse);

# greedy LT
def greedyLT(DG,k,p=0.1,mc=1000):
    S,spread,timeLapse,startTime=[],[],[],time.time();
    for i in range(k):
        bestSpread=0;
        node=None;
        for j in set(DG.nodes())-set(S):
            #print("j: ",j);
            newSpread=LT(DG,S+[j],p,mc);
            spreadSeedSet=LT(DG,S,p,mc);
            marginalSpread=newSpread-spreadSeedSet;
            #print("j: ",j,"newSpread: ",newSpread,"bestSpread: ",bestSpread,"seedset: ",S+[j]);
            if marginalSpread>bestSpread:
                bestSpread,node=newSpread,j;

        #print("node: ",node);
        if node!=None:
            S.append(node);
        spread.append(bestSpread);
        timeLapse.append(time.time()-startTime);
    return (S,spread,timeLapse);
#CELF seed selection
def CELF(DG,k,p=0.1,mc=1000):
    startTime=time.time();
    marginalGain=[IC(DG,[node],p,mc) for node in DG.nodes()];
    Q=sorted(zip(DG.nodes(),marginalGain),key=lambda x:x[1],reverse=True);
    S,spread,SPREAD=[Q[0][0]],Q[0][1],[Q[0][1]];
    Q,lookUps=Q[1:],[DG.number_of_nodes()];

    for i in range(k-1):
        check,nodeLookup=False,0;
        while (not check):
            node=Q[0][0];
            nodeLookup+=1;
            newSpread=IC(DG,S+[node],p,mc);
            Q[0]=(node,newSpread-spread);
            Q=sorted(Q,key=lambda x:x[1],reverse=True);
            check=(node==Q[0][0]);
        spread+=Q[0][1];
        S.append(Q[0][0]);
        SPREAD.append(spread);
        lookUps.append(nodeLookup);
        Q=Q[1:];
        timelapse=time.time()-startTime;
    return (S, SPREAD,timelapse, lookUps);

#CELF LT
def CELF_LT(DG,k,p=0.1,mc=1000):
    startTime=time.time();
    marginalGain=[LT(DG,[node],p,mc) for node in DG.nodes()];
    Q=sorted(zip(DG.nodes(),marginalGain),key=lambda x:x[1],reverse=True);
    S,spread,SPREAD=[Q[0][0]],Q[0][1],[Q[0][1]];
    Q,lookUps=Q[1:],[DG.number_of_nodes()];

    for i in range(k-1):
        check,nodeLookup=False,0;
        while (not check):
            node=Q[0][0];
            nodeLookup+=1;
            newSpread=IC(DG,S+[node],p,mc);
            Q[0]=(node,newSpread-spread);
            Q=sorted(Q,key=lambda x:x[1],reverse=True);
            check=(node==Q[0][0]);
        spread+=Q[0][1];
        S.append(Q[0][0]);
        SPREAD.append(spread);
        lookUps.append(nodeLookup);
        Q=Q[1:];
        timelapse=time.time()-startTime;
    return (S, SPREAD,timelapse, lookUps);


# CELF++
class NodeData:
    def __init__(self,node):
        self.node=node;
        self.mg1=0;
        self.flag=0;
        self.mg2=0;
        self.prevBest=None;
def CELFpp(DG,k,p=0.1,mc=1000):
    startTime = time.time();
    S,currentBest,node_data,heap,SPREAD=[],None,{},[],[];
    for node in DG.nodes():
        nodeDataObj=NodeData(node);
        nodeDataObj.mg1=IC(DG,[node],p,mc);
        nodeDataObj.prevBest=currentBest;
        nodeDataObj.mg2=IC(DG,[node,currentBest],p,mc)-IC(DG,[currentBest],p,mc) if currentBest else nodeDataObj.mg1;
        node_data[node]=nodeDataObj;
        heapq.heappush(heap, (-nodeDataObj.mg1, node));
        if currentBest is None or nodeDataObj.mg1 > node_data[currentBest].mg1:
            currentBest = node;
            print(currentBest);
    lastSeed=None;
    while len(S) < k:
        _,node=heapq.heappop(heap);
        nd=NodeData(node);
        if nd.flag==len(S):
            S.append(node);
            lastSeed=node;
            print(S);
            continue;
        if nd.prevBest==lastSeed:
            nd.mg1=nd.mg2;
        else:
            nd.mg1=IC(DG,S+[node],p,mc)-IC(DG,S,p,mc);
            nd.prevBest=currentBest;
            if currentBest:
                nd.mg2 = IC(DG, S + [currentBest, node], p, mc)- IC(DG, S + [currentBest], p, mc);
            else:
                nd.mg2=nd.mg1;
        nd.flag=len(S);
        node_data[node]=nd;############not there
        if currentBest is None or nd.mg1 > node_data[currentBest].mg1:
            currentBest = node;
        heapq.heappush(heap, (-nd.mg1, node));
    timeLapse=time.time()-startTime;
    return (S,timeLapse);


# zachary's karate club graph (testing)

degree_data = {
    0: 16, 1: 9, 2: 10, 3: 6, 4: 3, 5: 4, 6: 4, 7: 4, 8: 5,
    9: 2, 10: 3, 11: 1, 12: 2, 13: 5, 14: 2, 15: 2, 16: 2,
    17: 2, 18: 2, 19: 3, 20: 2, 21: 2, 22: 2, 23: 5, 24: 3,
    25: 3, 26: 2, 27: 4, 28: 3, 29: 4, 30: 4, 31: 6, 32: 12, 33: 17
}
G = nx.DiGraph();
G.add_nodes_from(degree_data.keys());
nodes = list(degree_data.keys());
for node, out_deg in degree_data.items():
    possible_targets = list(set(nodes) - {node});
    targets = random.sample(possible_targets, min(out_deg, len(possible_targets)));
    for target in targets:
        G.add_edge(node, target);
print(f"Created DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.");
print("spreadIC: ",IC(G,[0,5,9],0.1,1000));
print("spreadLT: ",LT(G,[0,5,9],0.1,1000));
greedy_output=greedy(G,4,0.1,10);
print("greedy output: " + str(greedy_output[0]));
greedy_output=greedyLT(G,4,0.1,10);
print("greedy output(LT): " + str(greedy_output[0]));
CELF_output=CELF(G,4,0.1,10);
print("CELF output: "+str(CELF_output[0]));
CELF_output=CELF_LT(G,4,0.1,10);
print("CELF output(LT): "+str(CELF_output[0]));
CELFpp_output=CELFpp(G,4,0.1,10);
print("CELFpp output: "+str(CELF_output[0]));









