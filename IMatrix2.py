#parties- bjp, tmc, leftfront
# opinion classes 1)support for bjp 2)support for tmc 3) support for leftfront 4)against bjp 5)against tmc 6)against leftfront
#represent this in an opinionated hypergraph.
# py -3.11 -m pip install fastjsonschema
import ast
import hypernetx as hnx
import pandas as pd
import matplotlib.pyplot as plt
import os
from polarityParty import positive_words, negative_words
from partyKeywords import party_keywords


folder_path = r"C:\Users\sahas\Downloads"

# List of CSV filenames to include
filenames = [
    "february_1.csv",
    "february_2.csv"
    "march_1.csv",
    "march_2.csv",
    "april_1.csv",
    "april_2.csv"
]

# Read and combine all files into one DataFrame
dfs = []
for file in filenames:
    file_path = os.path.join(folder_path, file)
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
                    #edges["against_tmc"].add(row["user_id"])
                    #edges["against_leftfront"].add(row["user_id"])
                elif party == "tmc":
                    edges["support_tmc"].add(row["user_id"])
                    # edges["against_bjp"].add(row['user_id'])
                    # edges["against_leftfront"].add(row["user_id"])
                elif party == "leftfront":
                    edges["support_leftfront"].add(row["user_id"])
                    # edges["against_bjp"].add(row['user_id'])
                    # edges["against_tmc"].add(row["user_id"])
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
hnx.draw(H, with_node_labels=False)
plt.title("Graph Visualisation");
plt.show();
