# The dataset for multi-party dialogue discourse parsing task
We mine the controversial topics in Reddit forum and proposed two new datasets: Multi-party Reddit Dialogue Link (MRDL) and Multi-party Reddit Dialogue Relation (MRDR). 
## Why reddit
The Reddit forums exhibit two text characteristics different from current datasets: <br />
(1) User utterances are usually long and have complex logic. <br />
(2) Related utterances are sometimes far away from each other in the dialogue due to the asynchronous nature of Reddit, for example other people insert into the conversations. 
<br />
![](https://raw.githubusercontent.com/AI0Research/MRDL-and-MRDR/main/explain/reddit2tree.png)
![](https://raw.githubusercontent.com/AI0Research/MRDL-and-MRDR/main/explain/example.png)
![](https://raw.githubusercontent.com/AI0Research/MRDL-and-MRDR/main/explain/dataset_fig1.png)

## Data analysis
MRDL consists of 28922 dialogues and 265078 utterances, its labels is whether two utterances has relationship, which is drawn from the actual reply in the forum. MRDR is a subset of MRDL that contains 15645 dialogues and 185823 utterances with human-labeled reply relationships as labels. Our datasets provide a means to evaluate the performance of multi-party dialogue systems in argumentative multi-party dialogues, which are currently lacking in existing datasets.
<br />
Our dataset naturally forms a graph network with multiple types of nodes, and the connections between these nodes contain important information. We random select a comment node, dialogue node, and user node respectively, along with their one-hop neighbors in the graph to construct subgraphs, which are visualized as follows. 
![](https://raw.githubusercontent.com/AI0Research/MRDL-and-MRDR/main/explain/dataset_statis.png)
<br />
The large red nodes at the center of each graph represent the initially selected nodes for the subgraph, comment nodes are orange, user nodes are blue, and dialogue nodes are grey. 
<br />
(a) shows the comment subgraph, which shows that a comment is published by a user, and has multiple reply comments, belonging to a specific dialogue. <br />

(b) shows the user subgraph, where a user makes comments, participates in multiple dialogues, and interacts with other users. The related user count may be greater than the comment count , because when the user makes a high-value comment, multiple users may reply to it.
<br />
(c) shows the dialogue subgraph, which contains users and comments. The edges between comments represent the reply relationships and the edges between users represent interaction relationships.
<br />
![](https://raw.githubusercontent.com/AI0Research/MRDL-and-MRDR/main/explain/graph.png)