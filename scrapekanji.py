import numpy as np
import codecs
import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareModel
import matplotlib.font_manager as mfm
from matplotlib.pyplot import figure
import matplotlib as mpl

with codecs.open('kanji.txt','r',encoding='utf-8') as file:
    kanji=file.read().split('\n')

with codecs.open('kokoro.txt', "r",encoding='ShiftJIS', errors='ignore') as file:
    kokoro=file.read().split('\n')

kokoro=''.join(kokoro)
n=len(kanji)
Count=np.zeros((n+1,n+1))
r=10

ni=0
for i in kanji:
    occur=[s for s, a in enumerate(kokoro) if a == i]
    nj=0
    selection=[]
    for loc in occur:
        selection.append(kokoro[loc-r:loc+r])
    selection=''.join(selection)
    for j in kanji:
        Count[ni,nj]+=selection.count(j)
        nj+=1
    print(i)
    ni+=1

rowsums=np.sum(Count,axis=1)
#np.sum(np.asarray([[1,1],[0,0]]),axis=1)
#np.asarray([[1,1],[0,0]])/np.expand_dims(np.asarray([1,2]),1)
Count2=Count[rowsums>0,rowsums>0]
Count3=Count2/np.expand_dims(rowsums[rowsums>0],1)
M=Count3+Count3.T
n=M.shape[0]
cut=3
relations=[]
for i in range(n):
    for j in range(i+1,n):
        if M[i,j]>cut:
            relations.append((kanji[i],kanji[j]))

negs=10
epcs=50
model = PoincareModel(relations, negative=negs,size=2)
model.train(epochs=epcs)

figure(figsize=(80, 60))
font_path = "kaiu.ttf"
mpl.rcParams["font.size"] = 4
prop = mfm.FontProperties(fname=font_path)
for i in list(model.kv.vocab.keys()):
    plt.scatter(model.kv.word_vec(i)[0],model.kv.word_vec(i)[1],s=0.1)
    plt.annotate(i,xy=(model.kv.word_vec(i)[0],model.kv.word_vec(i)[1]),fontproperties=prop)
plt.savefig(str(epcs)+'-'+str(negs)+'-'+str(cut)+'-'+str(r)+'.svg')
