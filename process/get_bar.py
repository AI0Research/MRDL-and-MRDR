from common_funcs import *
bar1 = {'<2':[1,2],'>=2 and <5':[2,5],">=5":[5,1000000]}
bar2 = {'=1':[1,2],'=2':[2,3],'=3':[3,4],'>=4':[4,1000]}
bar3 = {'<5':[1,5],'>=5 and <8':[5,8],'>=8 and <16':[8,16],'>=16':[16,10000]}
bar4 = {'<100':[1,100],'>=100 and <200':[100,200],'>=200':[200,10000]}
import pandas as pd
def get_range(x,bar1):
    for k,v in bar1.items():
        if x>=v[0] and x<v[1]:
            return k
    return 'no range'

def draw_distribution_bar(user_com_g_edges1,bar1,title,need_rotate=False):
    import matplotlib.pyplot as plt
    user_com_g_edges1['bar'] = user_com_g_edges1['count'].apply(lambda x: get_range(x,bar1))
    df_tmp = user_com_g_edges1.groupby('bar')['count'].count().reset_index()
    df_tmp['idx'] = df_tmp['bar'].apply(lambda x: [k[0] for idx,k in enumerate(bar1.items())].index(x))
    df_tmp = df_tmp.sort_values('idx')
    plt.figure(figsize=[16,10])
    color=['bisque','wheat','tan']
    color = ['skyblue','lightblue', 'slategray']
    plt.bar(x=df_tmp['bar'],height=df_tmp['count'],width = 0.6,color=color)
    # df_tmp.plot.bar(x='bar',y='count',title='The distribution of user\'s comment number')
    if need_rotate:
        plt.xticks(rotation=10)
    plt.title(title)
plt.rcParams.update({'font.size': 45})

user_com_g_edges1 = pd.DataFrame([len(user_com_g[n[0]]) for n in user_com_g.nodes(data=True) if n[1]['tp']=='author'],columns=['count'])
title='The distribution of\nuser\'s comment count'
draw_distribution_bar(user_com_g_edges1,bar1,title)

user_com_g_edges1 = pd.DataFrame([len(user_user_g[n[0]]) for n in user_user_g.nodes(data=True) if n[1]['tp']=='author'],columns=['count'])
title='The distribution of\nuser\'s related user count'
draw_distribution_bar(user_com_g_edges1,bar1,title)

user_com_g_edges1 = pd.DataFrame([len(com_com_g[n[0]]) for n in com_com_g.nodes(data=True)],columns=['count'])
title='The distribution of\ncomment\'s replies count'
draw_distribution_bar(user_com_g_edges1,bar2,title)

user_com_g_edges1 = pd.DataFrame([len(user_root_g[n[0]]) for n in user_root_g.nodes(data=True) if n[1]['tp']=='author'],columns=['count'])
title='The distribution of\nuser\'s related dialogue count'
draw_distribution_bar(user_com_g_edges1,bar1,title)

# comment_root
user_com_g_edges1 = pd.DataFrame([len(comment_root_g[n[0]]) for n in comment_root_g.nodes(data=True) if n[1]['tp']=='id'],columns=['count'])
title='The distribution of\ndialogue\'s comment count'
draw_distribution_bar(user_com_g_edges1,bar3,title,need_rotate=True)
# comment_root_len
arr = [np.mean([len(c) for c in comment_root_g[n[0]] if len(c)>10])  for n in comment_root_g.nodes(data=True) if n[1]['tp']=='id']
user_com_g_edges1 = pd.DataFrame(arr,columns=['count'])
title='The distribution of\ndialogue\'s average comment length'
draw_distribution_bar(user_com_g_edges1,bar4,title)
