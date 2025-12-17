#parties- bjp, tmc, leftfront
# opinion classes 1)support for bjp 2)support for tmc 3) support for leftfront 4)against bjp 5)against tmc 6)against leftfront
#represent this in an opinionated hypergraph.
# py -3.11 -m pip install fastjsonschema
import ast
import random
import hypernetx as hnx
import pandas as pd
import matplotlib.pyplot as plt
import os
from polarityParty import positive_words, negative_words
from partyKeywords import party_keywords
from HG_IM import (opinion_based_seed_selection,relevance_based_seed_selection,
                   polarity_aware_diffusion,LT_hypergraph,IC_hypergraph,greedyIC_hypergraph,CELF_IC_hypergraph
                   ,CELFPP_IC_hypergraph)

folder_path = r"C:\Users\sahas\Downloads"

# List of CSV filenames to include
filenames = [
    # "february_1.csv",
    # "february_2.csv",
    # "march_1.csv",
    # "march_2.csv",
    # "april_1.csv",
    "april_2.csv"
]

# Read and combine all files into one DataFrame
dfs = []
for file in filenames:
    file_path = os.path.join(folder_path, file)
    # file_path = "C:\Users\sahas\Downloads\mar_1.csv.xlsx"
    df_temp = pd.read_csv(file_path, dtype=str)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} total tweets from {len(filenames)} files.")


# party_keywords = {
#     "bjp": ["bjp", "modi", "amit shah", "indiawithmodi", "bengalwithnamo", "bengalwithbjp", "bengalwelcomesmodi", "pmmodi",
#             "narendramodi","mamtabanerjeekojaishriram","jaishriram","modiji","dhekiajuliwelcomesmodi","bjpgorbesonarbangla",
#             "atmanirbharpurvibharat","godimedia","bengalrejectsmodi","bengalrejectsbjp", "idiotpmmodi", "feku", "modiglobaldisaster","idiotpm",
#             "boycottbjp","chaiconspiracy","antifarmlaws","worldrenownedliar","modidhongihai","ram_card","modiinbengal","nirmalasitharaman",
#             "gretathunberg","modiwithfarmers","bengalwelcomespm","bjpgoonsattackingfarmers","rishiganga","pmoindia",
#             "wearewithutearakhand","prayforuttarakhand","poribortonyatra","rss","glaciar_burst","Chamoli","pmfloodfund",
#             "pmmodibusy","releasenodeepkaur","breakingnews","attack"],
#     "tmc": ["tmc", "mamata", "banerjee","mamtabanerjeekojaishriram","tmchataobanglabachao","didi","uttarakhanddisaster",
#             "murderermamata","duaresarkar","paraysamadhan","mamtabanerjee","suvenduadhikari","abhishekbanerjee","bhaipo","breakingnews",
#             "attack"],
#     "leftfront": ["left", "cpi", "marxist","leftlibgang"]
# }

opinion_classes = {    #each opinion class represents a hyperedge in the graph.
    1: "support_bjp",
    2: "support_tmc",
    3: "support_leftfront",
    4: "against_bjp",
    5: "against_tmc",
    6: "against_leftfront"
}

# positive_words = ["support", "vote for", "win", "love", "good", "indiawithmodi", "bengalwithnamo", "bengalwithbjp", "bengalwelcomesmodi",
#                   "mamtabanerjeekojaishriram","jaishriram","dhekiajuliwelcomesmodi","bjpgorbesonarbangla","atmanirbharpurvibharat",
#                   "modiwithfarmers","bengalwelcomespm"
#                   ] # for polarity
# negative_words = ["remove","down with","against","hate","bad","bjpdestroysdemocracy","godimedia","bengalrejectsmodi",
#                   "bengalrejectsbjp","fakepromises","idiotpmmodi","feku","modiglobaldisaster","idiotpm","boycottbjp",
#                   "chaiconspiracy","leftlibgang","antifarmlaws","worldrenownedliar","modidhongihai","ram_card",
#                   "mamtabanerjeekojaishriram","tmchataobanglabachao","uttarakhanddisaster","gretathunberg","murderermamata",
#                   "neverforgetneverforgive","bjpgoonsattackingfarmers","rishiganga","wearewithutearakhand","prayforuttarakhand",
#                   "propoganda","glaciar_burst","chamoli","flood","pmmodibusy","bhaipo","releasenodeepkaur"]

edges = {opinion_classes[i]: set() for i in range(1, 7)}
for _, row in df.iterrows():
    text = str(row['tweet']).lower() # extracting tweet and userid in each row
    for party, keywords in party_keywords.items():
        if any(k in text for k in keywords): # for identifying party
            if any(w in text for w in positive_words): #for identifying sentiments
                if party == "bjp":
                    edges["support_bjp"].add(row['user_id']) #users are nodes in this hypergraph
                    edges["against_tmc"].add(row["user_id"])
                    edges["against_leftfront"].add(row["user_id"])
                elif party == "tmc":
                    edges["support_tmc"].add(row["user_id"])
                    edges["against_bjp"].add(row['user_id'])
                    edges["against_leftfront"].add(row["user_id"])
                elif party == "leftfront":
                    edges["support_leftfront"].add(row["user_id"])
                    edges["against_bjp"].add(row['user_id'])
                    edges["against_tmc"].add(row["user_id"])
            elif any(w in text for w in negative_words):
                if party == "bjp":
                    edges["against_bjp"].add(row['user_id'])
                elif party == "tmc":
                    edges["against_tmc"].add(row["user_id"])
                elif party == "leftfront":
                    edges["against_leftfront"].add(row["user_id"])

H = hnx.Hypergraph(edges);
print("Nodes (users):", len(H.nodes))
print("Opinion categories (hyperedges):", len(H.edges))

#polarity_dict
polarity_dict = {}

for e in H.edges:                  # e = edge_name
    users = H.edges[e]             # set of users

    if e.startswith("support"):
        pol = 1
    elif e.startswith("against"):
        pol = -1
    else:
        continue

    for u in users:
        polarity_dict[(u, e)] = pol


# HG_IM
# print("Seed Set: ",opinion_based_seed_selection(H,5));
# T_prime = ["support_bjp", "against_tmc", "support_leftfront"]  # example topic subset
# r = {"support_bjp": 0.8, "against_tmc": 0.5, "support_leftfront": 0.6}

k = 5

# seed_set = relevance_based_seed_selection(H, T_prime, r, k)
# print("Seed Set:", seed_set)

# theta=0.6

#LT_HG

# print("LTspread: ",LT_hypergraph(H,seed_set,0.0,0.1,10))
# print("ICspread: ",IC_hypergraph(H,seed_set,0.5,10))

# S, spread, timeLapse = greedyIC_hypergraph(H,k,0.1,2)
# print("Selected seeds GIC: ", S)
# print("Spread after each iteration GIC: ", spread)
# print("Time elapsed GIC: ", timeLapse)


p = 0.01
mc = 10

# S, timeLapse, mean_spread = CELF_IC_hypergraph(H, k, p=p, mc=mc)
# S, timeLapse, mean_spread = greedyIC_hypergraph(H, k, p=p, mc=mc)
S, timeLapse, mean_spread = CELFPP_IC_hypergraph(H, k, p=p, mc=mc)

print("Final CELF++ seed set:", S)
print("Time lapse after each seed:", timeLapse)
print("Final mean spread:", mean_spread)

# print("Final seed set:", S)
# print("Time lapse:", timeLapse)
# print("Final mean spread:", mean_spread)


#IMPORTANT!!!!!
# activated=polarity_aware_diffusion(H,seed_set,polarity_dict,theta,rng=random.Random(42))
# print("Activated users:", activated)



# Draw
hnx.draw(H, with_node_labels=False)
plt.title("Graph Visualisation");
plt.show();


