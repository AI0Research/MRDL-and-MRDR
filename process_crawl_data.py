from common_funcs import *
import os
folder = 'dataset_MRD/'
comment_tree_all = {}
import json
for fname in os.listdir(folder):
    if 'res_freddit' in fname:
        f = open(folder+fname,encoding='utf8')
        for l in f:
            try:
                kv = json.loads(l.strip())
                for k,v in kv.items():
                    comment_tree_all[k] = v
            except:
                pass

folder = 'dataset_MRD/new_topic_dir/'
import json
for fname in os.listdir(folder):
    if 'res_freddit' in fname:
        f = open(folder+fname,encoding='utf8')
        for l in f:
            try:
                kv = json.loads(l.strip())
                for k,v in kv.items():
                    comment_tree_all[k] = v
            except:
                pass
