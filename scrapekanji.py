import numpy as np
import codecs
import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareModel

file=open('kanji.txt','r')
kanji=file.read().split('\n')
file.close()

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

M=Count+Count.T

relations=[]
for i in range(n+1):
    for j in range(i+1,n+1):
        if M[i,j]>0:
            relations.append((kanji[i],kanji[j]))


model = PoincareModel(relations, negative=1,size=2)
model.train(epochs=1)

plt.rcParams['font.family'] = 'IPAexGothic'
for i in list(model.kv.vocab.keys()):
    plt.scatter(model.kv.word_vec(i)[0],model.kv.word_vec(i)[1])
    plt.annotate(i,xy=(model.kv.word_vec(i)[0],model.kv.word_vec(i)[1]))
