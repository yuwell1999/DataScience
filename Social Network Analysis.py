import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_edge(1, 2)
# nx.draw_networkx(G)

# plt.show()

# G.add_nodes_from([3,4])

G.add_edge(3, 4)
G.add_nodes_from([(2, 3), (4, 1)])
# nx.draw_networkx(G)
# plt.show()

G = nx.krackhardt_kite_graph()
nx.draw_networkx(G)
plt.show()

print(nx.has_path(G, source=1, target=9))
print(nx.shortest_path(G, source=1, target=9))
print(nx.shortest_path_length(G, source=1, target=9))

# 输出所有最短路径
print(list(nx.shortest_simple_paths(G, source=1, target=9)))

print(nx.betweenness_centrality(G))
