from common_funcs import *

for_dialogue_test = 0
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
cur_tp = 'dialogue node center'
node_id = np.random.choice([n for n in user_user_g.nodes() if len(user_user_g[n])==10],1)[0]
# node_id = [n for n in user_user_g.nodes() if len(n)==3][0]
s_graph = big_graph.subgraph(list(big_graph[node_id])+[node_id])
G = s_graph
if cur_tp == 'user node center':
    node_id = np.random.choice([n for n in user_user_g.nodes() if len(user_user_g[n])==10],1)[0]
    s_graph = big_graph.subgraph(list(big_graph[node_id])+[node_id])
    G = s_graph
if cur_tp == 'dialogue node center':
    node_id = np.random.choice([n[0] for n in comment_root_g.nodes(data=True) if n[1]['tp']=='id' and len(comment_root_g[n[0]])==8],1)[0]
    node_list = list(big_graph[node_id])+[node_id]
    s_graph = big_graph.subgraph(node_list)
    if for_dialogue_test == 1:
        s_graph = big_graph.subgraph([n for n in node_list if big_graph.node[n]['tp']=='comment'])
    G = s_graph
if cur_tp == 'comment node center':
    node_id = np.random.choice([n[0] for n in com_com_g.nodes(data=True) if n[1]['tp']=='comment' and len(com_com_g[n[0]])>=5],1)[0]
    # node_id = [n for n in user_user_g.nodes() if len(n)==3][0]
    node_list = list(big_graph[node_id])+[node_id]
    s_graph = big_graph.subgraph(node_list)
    G = s_graph
center_node = node_id
node_colors = []
node_sizes = []
for n in list(G.nodes(data=True)):
    cur_color = ''
    cur_size = 0
    if n[1]['tp']=='comment':
        cur_color = 'orange'
        cur_size = 1000
    elif n[1]['tp']=='author':
        cur_color = 'lightblue'
        cur_size = 2000
    else:
        cur_color = 'grey'
        cur_size = 3000
    if n[0] == center_node:
        cur_color = 'red'
        cur_size = 6000

    node_colors.append(cur_color)
    node_sizes.append(cur_size)
label_arr = {}
user_id = 0
comment_id = 0
topic_id = 0
for n in list(G.nodes(data=True)):
    if n[1]['tp']=='author':
        user_id+=1
        label_arr[n[0]]='User'+str(user_id)+': '+n[0]
    elif n[1]['tp']=='comment':
        comment_id+=1
        label_arr[n[0]]='Comment'+str(comment_id)+":"+ ' '.join(n[0].split(' ')[:4])
    else:
        topic_id+=1
        label_arr[n[0]]='Dialogue'+str(topic_id)+': '+n[0]
# pos = nx.kamada_kawai_layout(G,scale=1)
pos = nx.kamada_kawai_layout(G,scale=0.5)
plt.figure(figsize=[25,15])
nx.draw(G, pos = pos,labels =label_arr,horizontalalignment='left',font_size=30,clip_on=True,node_color = node_colors,node_size=node_sizes,edge_color='grey',width=1,edge_alpha=1) 
# [1000 if not n == center_node else 5000 for n in G.nodes() ]
# nx.draw(G, pos = pos) 
# nx.draw_networkx_labels(G, pos=pos,labels =label_arr,horizontalalignment='left',font_size=25,clip_on=True)
# nx.draw_networkx_edges(G,pos = pos,width=0.1,alpha=0.01,edge_color='r')
# nx.draw_networkx_nodes(G,pos = pos,nodelist = list(G.nodes()),node_color = node_colors)
x_values, y_values = zip(*pos.values())
x_max = max(x_values)
x_min = min(x_values)
x_margin = (x_max - x_min) * 0.3
plt.xlim(x_min-x_margin, x_max + x_margin)
# plt.title(cur_tp)
plt.show()


# plt.gca().set_aspect('equal')
# plt.show()