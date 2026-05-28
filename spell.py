import sys
import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from tqdm import tqdm
import string

class LCSObject:
    def __init__(self, logTemplate='', logIDL=[], logParams={}):
        self.logTemplate = logTemplate
        self.logIDL = logIDL
        self.logParams = logParams

class Node:
    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()

class LogParser:
    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, rex=[], keep_para=True):
        self.path = indir
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.df_log = None

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1!=0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                result.insert(0, seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def SimpleLoopMatch(self, logClustL, seq):
        for logClust in logClustL:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq): continue
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        for i in range(idx, len(seq)):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if childn.logClust is not None:
                    constLM = [w for w in childn.logClust.logTemplate if w != '<*>']
                    if float(len(constLM)) >= self.tau * len(seq): return childn.logClust
                else: return self.PrefixTreeMatch(childn, seq, i + 1)
        return None

    def LCSMatch(self, logClustL, seq):
        maxLen, maxClust = -1, None
        set_seq, size_seq = set(seq), len(seq)
        for logClust in logClustL:
            set_t = set(logClust.logTemplate)
            if len(set_seq & set_t) < 0.5 * size_seq: continue
            lcs = self.LCS(seq, logClust.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen, maxClust = len(lcs), logClust
        return maxClust if float(maxLen) >= self.tau * size_seq else None

    def getTemplate(self, lcs, seq):
        if not lcs: return []
        lcs = lcs[::-1]
        retVal = []
        for token in seq:
            if lcs and token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else: retVal.append('<*>')
        return retVal

    def addSeqToPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = [w for w in newCluster.logTemplate if w != '<*>']
        for token in seq:
            if token not in parentn.childD:
                parentn.childD[token] = Node(token=token, templateNo=1)
            else: parentn.childD[token].templateNo += 1
            parentn = parentn.childD[token]
        parentn.logClust = newCluster

    def removeSeqFromPrefixTree(self, rootn, cluster):
        parentn = rootn
        seq = [w for w in cluster.logTemplate if w != '<*>']
        for token in seq:
            if token in parentn.childD:
                matchedNode = parentn.childD[token]
                if matchedNode.templateNo <= 1:
                    del parentn.childD[token]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def parse(self, logname):
        starttime = datetime.now()
        self.logname = logname
        self.load_data()
        rootNode, logCluL = Node(), []
        
        for _, line in tqdm(self.df_log.iterrows(), total=len(self.df_log), desc="Spell Parsing"):
            logID = line['LineId']
            logmessageL = list(filter(lambda x: x != '', re.split(r'[\s=:,()]', self.preprocess(line['Content']))))
            constLogMessL = [w for w in logmessageL if w != '<*>']
            
            matchCluster = self.PrefixTreeMatch(rootNode, constLogMessL, 0)
            if matchCluster is None: matchCluster = self.SimpleLoopMatch(logCluL, constLogMessL)
            if matchCluster is None: matchCluster = self.LCSMatch(logCluL, logmessageL)

            if matchCluster is None:
                newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)
            else:
                newT = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate), matchCluster.logTemplate)
                if ' '.join(newT) != ' '.join(matchCluster.logTemplate):
                    self.removeSeqFromPrefixTree(rootNode, matchCluster)
                    matchCluster.logTemplate = newT
                    self.addSeqToPrefixTree(rootNode, matchCluster)
                matchCluster.logIDL.append(logID)

        if not os.path.exists(self.savePath): os.makedirs(self.savePath)
        self.outputResult(logCluL)
        print(f'Parsing done. [Time: {datetime.now() - starttime}]')

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        target = self.logname if os.path.isabs(self.logname) else os.path.join(self.path, self.logname)
        log_messages = []
        with open(target, 'r') as fin:
            for line in fin:
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line).replace('  ', ' ')
                try:
                    match = regex.search(line.strip())
                    if match: log_messages.append([match.group(h) for h in headers])
                except: pass
        self.df_log = pd.DataFrame(log_messages, columns=headers)
        self.df_log.insert(0, 'LineId', range(1, len(self.df_log) + 1))

    def preprocess(self, line):
        for currentRex in self.rex: line = re.sub(currentRex, '<*>', line)
        return line

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0: regex += splitters[k].replace(' ', r'\s+')
            else:
                header = splitters[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
        return headers, re.compile('^' + regex + '$')

    def outputResult(self, logClustL):
        templates = [0] * len(self.df_log)
        ids = [0] * len(self.df_log)
        for i, clust in enumerate(logClustL):
            t_str = ' '.join(clust.logTemplate)
            eid = i + 1
            for lid in clust.logIDL:
                templates[lid-1] = t_str
                ids[lid-1] = eid
        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        self.df_log.to_csv(os.path.join(self.savePath, 'spell_structured.csv'), index=False)