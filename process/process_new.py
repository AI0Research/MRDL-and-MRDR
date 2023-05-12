from common_funcs import *
import urllib.request
import sys
import json
import ssl
from get_comment_reddit import get_comment_info
import pandas as pd
import praw
import sys
ssl._create_default_https_context = ssl._create_unverified_context
# input the user id and pwd
reddit = praw.Reddit()
parent_topic = []
f1 = open('res_freddit_'+sys.argv[1],'w')
target_set = set(['author','utc_datetime_str','body'])
for l in open(sys.argv[1]):
#    try:
   cid=l.strip()
   cmnt = reddit.comment(id=cid)
   topicid = str(cmnt.submission)
   f1.write(json.dumps(get_comment_info(topicid,reddit))+"\n")
   f1.flush()           
#    except:
#        print(l)
f1.flush()

