
import re
import os
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm

class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        self.logIDL = logIDL if logIDL is not None else []

class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        self.childD = childD if childD is not None else dict()
        self.depth = depth
        self.digitOrtoken = digitOrtoken

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4, 
                 maxChild=100, rex=[], keep_para=True):
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None
        seqLen = len(seq)
        if seqLen not in rn.childD: return retLogClust
        parentn = rn.childD[seqLen]
        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen: break
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else: return retLogClust
            currentDepth += 1
        return self.fastMatch(parentn.childD, seq)

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]
        parentn = firtLayerNode
        currentDepth = 1
        for token in logClust.logTemplate:
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0: parentn.childD = [logClust]
                else: parentn.childD.append(logClust)
                break
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else: parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else: parentn = parentn.childD['<*>']
                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else: parentn = parentn.childD['<*>']
            else: parentn = parentn.childD[token]
            currentDepth += 1

    def seqDist(self, seq1, seq2):
        simTokens = 0
        numOfPar = 0
        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2: simTokens += 1 
        return float(simTokens) / len(seq1), numOfPar

    def fastMatch(self, logClustL, seq):
        maxSim, maxNumOfPara, maxClust = -1, -1, None
        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim, maxNumOfPara, maxClust = curSim, curNumOfPara, logClust
        return maxClust if maxSim >= self.st else None

    def getTemplate(self, seq1, seq2):
        retVal = []
        for i, word in enumerate(seq1):
            retVal.append(word if word == seq2[i] else '<*>')
        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        for i, logClust in enumerate(logClustL):
            template_str = ' '.join(logClust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                log_templates[logID - 1] = template_str
                log_templateids[logID - 1] = eid

        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        if not os.path.exists(self.savePath): os.makedirs(self.savePath)
        save_name = os.path.basename(self.logName)
        self.df_log.to_csv(os.path.join(self.savePath, save_name + '_structured.csv'), index=False)

    def parse(self, logName):
        self.logName = logName
        self.load_data()
        rootNode = Node()
        logCluL = []
        for idx, line in tqdm(self.df_log.iterrows(), total=len(self.df_log), desc="Parsing Logs"):
            logmessageL = self.preprocess(line['Content']).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[line['LineId']])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(line['LineId'])
                matchCluster.logTemplate = newTemplate
        self.outputResult(logCluL)

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        log_messages = []
        target_path = self.logName if os.path.isabs(self.logName) else os.path.join(self.path, self.logName)
        with open(target_path, 'r') as fin:
            for line in fin:
                try:
                    match = regex.search(line.strip())
                    if match:
                        log_messages.append([match.group(header) for header in headers])
                except Exception: pass
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
            if k % 2 == 0:
                # FIX: Use literal replacement to avoid 'bad escape \s' in Python 3.12
                regex += splitters[k].replace(' ', r'\s+')
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        return headers, re.compile('^' + regex + '$')