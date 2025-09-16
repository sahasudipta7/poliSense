import matplotlib.pyplot as plt;
import networkx as nx;
import pandas as pd;

DG=nx.DiGraph();

DG.add_nodes_from([0,1,2,3]);
DG.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (3, 0), (3, 2)]);

#draw

pos=nx.spring_layout(DG);
nx.draw(DG,pos,with_labels=True,node_size=700,node_color="blue",edge_color="green");
plt.title("Graph Visualisation");
plt.show();

#centrality

deg_cen=nx.degree_centrality(DG);
bet_cen=nx.betweenness_centrality(DG);
eig_cen=nx.eigenvector_centrality(DG,max_iter=1000);

# printing as a table
df = pd.DataFrame({
    "Node": list(DG.nodes()),
    "Degree Centrality": [deg_cen[n] for n in DG.nodes()],
    "Betweenness Centrality": [bet_cen[n] for n in DG.nodes()],
    "Eigenvector Centrality": [eig_cen[n] for n in DG.nodes()]
}).set_index("Node").round(3)

print(df)

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
