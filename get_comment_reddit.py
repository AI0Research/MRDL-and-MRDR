from common_funcs import *
import praw

def get_comment_info(topicid,reddit):
    comment_tree = {}
    submission = reddit.submission(topicid)

    submission.comments.replace_more(limit=None)
    import copy
    comment_queue = submission.comments[:]  # Seed with top-level
    root_list = copy.deepcopy(comment_queue)

    while comment_queue:
        comment = comment_queue.pop(0)
        try:
            comment_tree[comment.id] = {'child': [c1.id for c1 in comment.replies[:]],'is_root':0,
                                    'body':comment.body,'author':comment.author.name}
        except:
            pass

        for c in comment.replies:
            try:
                if len(c.body) >= 5:
                    comment_tree[c.id] = {'child': [c1.id for c1 in c.replies[:]],'is_root':0,
                                            'body':c.body,'author':c.author.name}
                    comment_queue.append(c)
            except:
                pass

    import copy
    for  c in root_list:
        c_avg_len = 0
        all_len = 0
        c_node_num = 1
        c_long_node_num =0
        c_arr = copy.deepcopy(comment_tree.get(c.id,{}).get('child',[]))
        if len(c_arr) == 0:
            continue
        while len(c_arr) > 0:
            c1 = c_arr.pop(0)
            body_len = len(comment_tree.get(c1,{}).get('body',""))
            all_len += body_len
            c_node_num += 1
            if body_len > 10:
                c_long_node_num +=1
            c_arr += comment_tree.get(c1,{}).get('child',[])
        if c_node_num >= 4:
            comment_tree[c.id]['node_num'] = c_node_num
            comment_tree[c.id]['long_node_num'] = c_long_node_num
            comment_tree[c.id]['avg_len'] = all_len/c_node_num
            comment_tree[c.id]['is_root'] = 1
    return comment_tree
