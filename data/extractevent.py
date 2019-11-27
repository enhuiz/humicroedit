# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:13:05 2019

@author: 24707
"""
#%%
from stanfordcorenlp import StanfordCoreNLP
#from nltk.parse.corenlp import CoreNLPDependencyParser
import logging
import json
import sys
import os
import numpy as np
import itertools

try:
    sys.path.append('.')
    from humicroedit.datasets.humicroedit import HumicroeditDataset
except ImportError as e:
    print(e)
    print('Please run under the root dir, but not {}.'.format(os.getcwd()))
    exit()
ds = HumicroeditDataset('data/humicroedit/task-1', 'train')
#samples,id,sentence,meangrade
samples=ds.samples

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

def pattern_match(D,posd1,pwi,centerverb,pattern,n1=['NN','NNS','NNP','NNPS'],v1=['VB','VBP','VBD','VBZ','VBN','VBG'],a1=['JJS','JJR','JJ']):
    '''
    D:extracted dependency, [dependency,ind1,ind2]
    posd:positive dependency of a certain pattern  [pos,dependency,pos]
    pwi:index as key, (word,pos) as value
    '''
    E=[]
    flag=0 #whether match with this pattern
    for dd in D:
        if dd[0]!='ROOT':
            sel1=np.where(posd1[:,1]==dd[0]) #dependecny match
            sel2=np.where(np.any([posd1[:,0]==pwi[dd[1]][1],posd1[:,0]==pwi[dd[2]][1]])) #first word pos match
            sel3=np.where(np.any([posd1[:,2]==pwi[dd[2]][1],posd1[:,2]==pwi[dd[1]][1]])) #second word pos match
            selind=np.intersect1d(sel1,np.intersect1d(sel2,sel3))
            if selind.size!=0:
                dd2append=[[dd[1],pwi[dd[1]][0],pwi[dd[1]][1]],dd[0],[dd[2],pwi[dd[2]][0],pwi[dd[2]][1]]]
                E.append(dd2append)
                #remove all the relevent relaions in posd
                selitem=np.squeeze(posd1[selind])
                rmind1=np.where(posd1[:,1]==dd[0])
                if selitem[0] in n1: #the first word is noun
                    rmind2=np.where(np.in1d(posd1[:,0], n1))[0]
                    if selitem[2] in n1: #the second word is noun
                        rmind3=np.where(np.in1d(posd1[:,2],n1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in v1: #the second word is verb
                        rmind3=np.where(np.in1d(posd1[:,2],v1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in a1:
                        rmind3=np.where(np.in1d(posd1[:,2],a1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                elif selitem[0] in v1:
                    rmind2=np.where(np.in1d(posd1[:,0], v1))[0]
                    if selitem[2] in n1:
                        rmind3=np.where(np.in1d(posd1[:,2],n1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in v1:
                        rmind3=np.where(np.in1d(posd1[:,2],v1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in a1:
                        rmind3=np.where(np.in1d(posd1[:,2],a1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                elif selitem[0] in a1:
                    rmind2=np.where(np.in1d(posd1[:,0], a1))[0]
                    if selitem[2] in n1:
                        rmind3=np.where(np.in1d(posd1[:,2],n1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in v1:
                        rmind3=np.where(np.in1d(posd1[:,2],v1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                    elif selitem[2] in a1:
                        rmind3=np.where(np.in1d(posd1[:,2],a1))[0]
                        rmind=np.intersect1d(rmind1,np.intersect1d(rmind2,rmind3))
                posd1=np.delete(posd1,rmind,axis=0)
        if posd1.size==0:
            flag=1
            return flag,E
    if posd1.size==0:
        flag=1
                    
                
    return flag,E



if __name__ == '__main__':
    sNLP = StanfordNLP()
    posd={}
    n1=['NN','NNS','NNP','NNPS']
    relation1=['nsubj']
    a1=['JJS','JJR','JJ']
    posd1=[n1,relation1,a1]   #n1-nsubj-a1
    posd1=list(itertools.product(*posd1))
    v1=['VB','VBP','VBD','VBZ','VBN','VBG']
    posd2=[n1,relation1,v1]  #n1-subj-v1
    posd2=list(itertools.product(*posd2))
    relation2=['nsubjpass']
    posd3=[n1,relation2,v1]
    posd3=list(itertools.product(*posd3)) #n1,nsubjpass,v1
    relation3=['dobj']
    posd4=[v1,relation3,n1]
    posd4=list(itertools.product(*posd4)) #v1,dobj,n2
    relation4=['xcomp']
    posd5=[v1,relation4,v1]
    posd5=list(itertools.product(*posd5)) #v1-xcomp-v2
    posd6=[v1,relation4,a1]
    posd6=list(itertools.product(*posd6)) #v1-xcomp-a1
    posd7=[v1,relation4,n1]
    posd7=list(itertools.product(*posd7)) #v1-xcomp-n1
    relation5=['iobj']
    posd8=[v1,relation5,n1]
    posd8=list(itertools.product(*posd8)) #v1-iobj-n1
    posd9=[n1,relation3,n1]
    posd9=list(itertools.product(*posd9)) #n1-dobj-n2
    relation6=['cop']
    posd10=[a1,relation6,v1]
    posd10=list(itertools.product(*posd10)) #a1-cop-be
    posd11=[n1,relation6,v1]
    posd11=list(itertools.product(*posd11))  #n1-cop-be
    relation7=['nmod']
    posd12=[v1,relation7,n1]
    posd12=list(itertools.product(*posd12)) #v1-nmod-n1
    posd13=[n1,relation7,n1]
    posd13=list(itertools.product(*posd13)) #n1-nmod-n1
    relation8=['case']
    p1=['IN']
    posd14=[n1,relation8,p1]
    posd14=list(itertools.product(*posd14)) #n1-case-p1
    posd['s-v']=np.asarray(posd2)
    posd['s-v-o']=np.asarray(posd2+posd4) #n1-nsubj-v1-dobj-v2
    posd['s-v-a']=np.asarray(posd2+posd6)#n1-nsubj-v1-xcomp-a
    posd['s-v-o-o']=np.asarray(posd2+posd8+posd9)#n1-nsubj-(v1-iobj-n2)-dobj-n3
    posd['s-be-a']=np.asarray(posd1+posd10) #n1-nsubj-a1-cop-be
    posd['s-v-be-a']=np.asarray(posd2+posd6+posd10)#n1-nsubj-v1-xcomp-a1-cop-be
    posd['s-v-be-o']=np.asarray(posd2+posd7+posd11)  #n1-nsubj-v1-xcomp-n2-cop-be
    posd['s-v-v-o']=np.asarray(posd2+posd5+posd4) #n1-nsubj-v1-xcomp-v2-dobj-n2
    posd['s-v-v']=np.asarray(posd2+posd5) #n1-nsubj-v1-xcomp-v2
    posd['s-be-a-p-o']=np.asarray(posd1+posd10+posd12+posd14) #(n1-nsubj-a1-cop-be)-nmod-n2-case-p1
    posd['s-v-p-o']=np.asarray(posd2+posd12+posd14)  #n1-nsubj-v1-nmod-n2-case-p1
    posd['s-v-o-p-o']=np.asarray(posd2+posd4+posd13+posd14)  #(n1-nsubj-v1-dobj-n2)-nmod-n3-case-p1
    posd['spass-v']=np.asarray(posd3)  #n1-nsubjpass-v1
    posd['spass-v-p-o']=np.asarray(posd3+posd12+posd14) #n1-nsubjpass-v1-nmod-n2-case-p1
    Event={}   
    for samp in samples:
        idx=samp[0]
        text=samp[1]
        meangrade=samp[2]
        pos=sNLP.pos(text)
        wtokens=sNLP.word_tokenize(text)
        ind2token={i+1:w for i,w in enumerate(wtokens)}#index starting from 1
#        token2ind={y:x for x,y in ind2token.items()}
        dp=sNLP.dependency_parse(text)
        centerverb=ind2token[dp[0][2]]
        pwi={i+1:w for i,w in enumerate(pos)}
        
        for pattern in posd.keys():
            flag,E=pattern_match(dp,posd[pattern],pwi,centerverb,pattern)
            if flag==1:
                Event[idx]={'text':text,'meangrade':meangrade,'pattern':pattern,'eventuality':E}
                
import json

with open('event.json', 'w') as fp:
    json.dump(Event, fp)        
        
        
        
        
#    text = 'The dog barks.'#Barack Obama was born in Hawaii. He was elected president in 2008.'
#    print ("Annotate:", sNLP.annotate(text))
#    print ("POS:", sNLP.pos(text))
#    print ("Tokens:", sNLP.word_tokenize(text))
#    print ("NER:", sNLP.ner(text))
#    print ("Parse:", sNLP.parse(text))
#    print ("Dep Parse:", sNLP.dependency_parse(text))

#%%
#import os
#from nltk.parse.corenlp import CoreNLPDependencyParser
#
#modelpath='C:\\Users\\24707\\OneDrive - HKUST Connect\\courses\\comp5222\\stanford-english-corenlp-2018-10-05-models\\edu\\stanford\\nlp\\models\\lexparser\\englishPCFG.ser.gz'
##dep_parser=CoreNLPDependencyParser(model_path=modelpath)
#dep_parser=CoreNLPDependencyParser(url='http://localhost:9000')
#print ([parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")])





#%%
import sys
import os
import pandas as pd
import stanfordnlp
try:
    sys.path.append('.')
    from humicroedit.datasets.humicroedit import HumicroeditDataset
except ImportError as e:
    print(e)
    print('Please run under the root dir, but not {}.'.format(os.getcwd()))
    exit()
sentences=pd.read_csv('humicroedit/task-1/train.csv')

ds = HumicroeditDataset('humicroedit\\task-1', 'train')


#stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()

sentences=pd.read_csv('humicroedit/task-1/train.csv')

ds = HumicroeditDataset('data/humicroedit/task-1', 'train')