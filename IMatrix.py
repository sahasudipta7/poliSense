#parties- bjp, tmc, leftfront
# opinion classes 1)support for bjp 2)support for tmc 3) support for leftfront 4)against bjp 5)against tmc 6)against leftfront
#represent this in an opinionated hypergraph.
# py -m pip install fastjsonschema
import ast
import hypernetx as hnx
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\sahas\Downloads\february_1.csv")  #uploaded file

party_keywords = {
    "bjp": ["bjp", "modi", "amit shah"],
    "tmc": ["tmc", "mamata", "banerjee"],
    "leftfront": ["left", "cpi", "marxist"]
}

opinion_classes = {    #each opinion class represents a hyperedge in the graph.
    1: "support_bjp",
    2: "support_tmc",
    3: "support_leftfront",
    4: "against_bjp",
    5: "against_tmc",
    6: "against_leftfront"
}

positive_words = ["support", "vote for", "win", "love", "good"] # for polarity
negative_words = ["remove", "down with", "against", "hate", "bad"]

edges = {opinion_classes[i]: set() for i in range(1, 7)}

for _, row in df.iterrows():
    text = str(row['tweet']).lower() # extracting tweet and userid in each row
    for party, keywords in party_keywords.items():
        if any(k in text for k in keywords): # for identifying party
            if any(w in text for w in positive_words): #for identifying sentiments
                if party == "bjp":
                    edges["support_bjp"].add(row['user_id']) #users are nodes in this hypergraph
                elif party == "tmc":
                    edges["support_tmc"].add(row["user_id"])
                elif party == "leftfront":
                    edges["support_leftfront"].add(row["user_id"])
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

# Draw
hnx.draw(H, with_node_labels=True)
plt.title("Graph Visualisation");
plt.show();
