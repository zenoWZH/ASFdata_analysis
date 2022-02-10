#import modin.pandas as pd
import os
import time
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gc

to_path = './entropynet_data/emails/'
if not os.path.exists(to_path):
    os.makedirs(to_path)
#e_path = '/mnt/data0/lkyin/monthly_data/emails/'
e_path = '../author_data/emails/'
projects = os.listdir(e_path)
projects.sort(key= lambda x:int(x.split('__')[1].split('.')[0]))
gx_socialnets = []
for project in tqdm(projects):
    # Here period mean individual authors
    social_net = {}
    emailID_to_author = {}
    project_name, period = project.replace('.csv', '').split('__')
    if int(period) == 0 :
        continue
    fnames = [project_name+'__'+str(int(period))+'.csv']
    
    projdfname = project_name+'__0.csv'
    #if not os.path.exists(projdfname):
    #    continue
    projdf = pd.read_csv(e_path+projdfname)
    for index, row in projdf.iterrows():
        message_id = row['message_id'].strip()
        # print(row['dealised_author_full_name'])
        prev_prev_author = row['dealised_author_full_name']
        emailID_to_author[message_id] = prev_prev_author

    for fname in fnames:
        if not os.path.exists(e_path+fname):
            continue
        df = pd.read_csv(e_path+fname)
        df.query('is_bot == False', inplace=True)
        df = df[df['dealised_author_full_name'].notna()]
        
        
        # raise KeyError
        for index, row in df.iterrows():
            message_id = row['message_id']
            references = row['references']
            prev_prev_author = row['dealised_author_full_name']

            # ignores if this email does not to previous emails
            if pd.isna(references) or references == 'None':
                continue

            references = [r.strip() for r in references.replace('\n', ' ').replace('\t', ' ').split(' ') if r.strip()]

            # deal with the issue that a line breaker exists in message_id:
            # e.g., <4\n829AB62.6000302@apache.org>
            new_refs = set()
            for i in range(len(references)-1):
                if '<' in references[i] and '>' not in references[i] and '<' not in references[i+1] and '>' in references[i+1]:
                    new_refs.add(references[i] + references[i+1])
            for r in references:
                if '<' in r and '>' in r:
                    new_refs.add(r)
            prev_prev_author = None
            references = new_refs
            for reference_id in references:
                if reference_id not in emailID_to_author:
                    continue
                prev_author = emailID_to_author[reference_id]
                # if it's the same person, continue
                if prev_author == prev_prev_author:
                    continue
                if not(prev_prev_author):
                    prev_prev_author = prev_author
                    continue
                if prev_prev_author not in social_net:
                    social_net[prev_prev_author] = {}
                if prev_author not in social_net:
                    social_net[prev_author] = {}

                # if node B replies node A, it means B sends signal to A
                if prev_author not in social_net[prev_prev_author]:
                    social_net[prev_prev_author][prev_author] = {}
                    social_net[prev_prev_author][prev_author]['weight'] = 0
                social_net[prev_prev_author][prev_author]['weight'] += 1

    gc.collect()
    
    #save as directed graph
    g = nx.DiGraph(social_net)
    # add disconnected nodes
    g.add_nodes_from(social_net.keys())
    nx.write_edgelist(g, to_path + '{}__{}.edgelist'.format(project_name, str(period)), delimiter='##', data=["weight"])
    gx_socialnets.append(g)