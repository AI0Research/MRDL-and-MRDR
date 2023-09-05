from common_funcs import *
def remove_dup_comment(edus):
    e_map = {}
    edus1 = []
    for e in edus:
        e_arr = [e_sub.replace('>','').strip() for e_sub in e['text'].split('\n')]
        e_arr1 = []
        for e_sub in e_arr:
            is_bad_e = 0
            if not e_sub in e_map:
                for e_sub1 in e_map.keys():
                    if e_sub in e_sub1:
                        is_bad_e = 1
                e_map[e_sub] = 1
            else:
                is_bad_e = 1
            if is_bad_e == 0:
                e_arr1.append(e_sub)
        txt = '\n'.join(e_arr1)
        edus1.append({'text':txt,'speaker':e['speaker']})
    return edus1


gpt_test = []
tp_dct = {}
all_comment = []
for idx,ss1 in enumerate(target_root.items()):
    idx = 0
    single_comment = {}
    single_comment['edus'] = []
    edus_arr = []
    single_comment['relations'] = []
    single_comment['id'] = idx
    cm_arr = [ss1[0]]
    level = 0
    f_arr = []
    while len(cm_arr) > 0:
        c = cm_arr.pop(0)
        if comment_tree_all.get(c,{}).get('body','')!='' and comment_tree_all.get(c,{}).get('author','') !='':
            f_arr.append(c)
        child = comment_tree_all.get(c,{}).get('child',[])
        for cd in child:
            cm_arr.append(cd)
    for a in f_arr:
        text1 = comment_tree_all.get(a,{}).get('body','')
        text2 = comment_tree_all.get(a,{}).get('author','')
        edus_arr.append({'text':text1,'speaker':text2})
        child = comment_tree_all.get(a,{}).get('child',[])
        
        for cd in child:
            if cd in f_arr:
                single_comment['relations'].append({'y': f_arr.index(cd), 'x': f_arr.index(a),"type": ""})
    single_comment['edus'] = remove_dup_comment(edus_arr)
#     single_comment['edus'] = edus_arr
    all_comment.append(single_comment)

    for r in single_comment['relations']:
        child = single_comment['edus'][r['y']]['text']
        fath = single_comment['edus'][r['x']]['text']
        gpt_test.append({'child':child,'father':fath,'id':idx})    
    idx += 1
def write_dataset(all_comment,fname):
    all_comment_str = json.dumps(all_comment)
    d_len = len(all_comment)
    f1 = open('train_'+fname,'w',encoding='utf8')
    f1.write(json.dumps(all_comment[:d_len-2000]).replace('\n',''))
    f1.flush()

    f1 = open('test_'+fname,'w',encoding='utf8')
    f1.write(json.dumps(all_comment[d_len-2000:]).replace('\n',''))
    f1.flush()

    f1 = open('dev_'+fname,'w',encoding='utf8')
    f1.write(json.dumps(all_comment[d_len-2000:]).replace('\n',''))
    f1.flush()