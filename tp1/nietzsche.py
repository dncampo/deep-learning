from collections import Counter
import numpy as np
from sklearn import decomposition
import pylab as pl

dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/'

lines=[]
f=open(dataDir+"nietzsche.txt")
char=f.read(1)
line=""
while char:
    if char=='\n':
        char=" "
    if char!='.' and char!='?' and char!='!': # mienras no sea fin de oración
        line+=char
    else:
        line+=char
        lines.append(line)
        line=""
    char=f.read(1)
    #if len(lines)==10: # borrar esto
    #    break


# Frecuencias de aparición
wfreq=Counter()
# aca se pueden considerar otros signos de puntuación como palabras o descartarlos
punctuation=[',',':',';','?','.','!','--','"','(',')','=','_'] # el guion '-' define palabras nuevas, no lo agrego acá
lines_split=[] # redefino la oración en palabras (por los signos de puntuación lo hago acá)
for l in lines:
    l_split=[]
    for p in punctuation:
        pcount=0
        pind=l.find(p)
        while pind>-1:
            pcount+=1
            pind=l.find(p,pind+1)
            l_split.append(p)
        if pcount:
            wfreq[p]+=pcount
                
            l=l.replace(p,' ')
    for w in l.split(" "):
        w=w.lower() # podría ser una palabra distinta la que comience la oracion
        if w!="":
            wfreq[w]+=1
            l_split.append(w)
    lines_split.append(l_split)
# las mas comunes
wcomunes=[]
for w in wfreq.most_common()[0:1000]:
    wcomunes.append(w[0])
# las mas raras # por ahora todas estas tienen solo 1 repeticion
wraras=[]
for w in wfreq.most_common()[-1000:]:
    wraras.append(w[0])

# Transformar las oraciones en función de wcomunes y wraras
nietzscheDataset=np.zeros([len(lines),2000])
for (k,l) in enumerate(lines_split):
    # contabilizar cada palabra y asignarla a los vectores wcomunes y wraras.
    for w in l:
        if w in wcomunes:
            nietzscheDataset[k,wcomunes.index(w)]+=1/len(l) # frecuencia
        if w in wraras:
            nietzscheDataset[k,1000+wraras.index(w)]+=1/len(l) 
    # freq inversa
    nietzscheDataset[k,1000:]=1-nietzscheDataset[k,1000:] 
        

# PCA
pca = decomposition.PCA(n_components=2)
pca.fit(nietzscheDataset)
X = pca.transform(nietzscheDataset)
pl.scatter(X[:, 0], X[:, 1]) 
pl.show()


