from common_funcs import *
import networkx as nx
user_com_g = nx.Graph()
user_root_g = nx.Graph()
comment_root_g = nx.Graph()
user_user_g = nx.Graph()
com_com_g = nx.Graph()
big_graph = nx.Graph()
for k,v in comment_tree_sub.items():
    user_com_g.add_edge(v['author'],v['body'])
    big_graph.add_edge(v['author'],v['body'])
    for c in v['child']:
        if c in comment_tree_sub:
            big_graph.add_edge(v['author'],comment_tree_sub[c]['author'])
            big_graph.add_edge(v['body'],comment_tree_sub[c]['body'])
            big_graph.add_edge(v['author'],comment_tree_sub[c]['ancestor'])
            big_graph.add_edge(v['ancestor'],comment_tree_sub[c]['body'])

            user_user_g.add_edge(v['author'],comment_tree_sub[c]['author'])
            com_com_g.add_edge(v['body'],comment_tree_sub[c]['body'])
            user_root_g.add_edge(v['author'],comment_tree_sub[c]['ancestor'])
            comment_root_g.add_edge(v['ancestor'],comment_tree_sub[c]['body'])
attrs ={}
for k,v in comment_tree_all.items():
    attrs[v['body']] = {'tp':'comment'}
    attrs[v['author']] = {'tp':'author'}
    attrs[k] = {'tp':'id'}
       
nx.set_node_attributes(user_com_g,{k:v for k,v in attrs.items() if k in user_com_g})
nx.set_node_attributes(user_root_g,{k:v for k,v in attrs.items() if k in user_root_g})
nx.set_node_attributes(user_user_g,{k:v for k,v in attrs.items() if k in user_user_g})
nx.set_node_attributes(com_com_g,{k:v for k,v in attrs.items() if k in com_com_g})
nx.set_node_attributes(comment_root_g,{k:v for k,v in attrs.items() if k in comment_root_g})
nx.set_node_attributes(big_graph,{k:v for k,v in attrs.items() if k in big_graph})

print(len([(k,v) for k,v in comment_tree_sub.items() if v.get('is_root',0)==1]))
print(len(comment_tree_sub))
