def accuracy(a,b):
    acc=0
    for n in range(0,len(a)):
        if a[n]==b[n]:
            acc+=1
    return 100*acc/len(a)
    
