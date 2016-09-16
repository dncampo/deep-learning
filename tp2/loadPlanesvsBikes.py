def loadPlanesvsBikes(dataDir):

    imFiles=[]
    for d in os.walk([dataDir,'motorbikes/']):
        for f in d[2]:
            if ".directory" not in f:
                imFiles.append(d[0]+"/"+f)
    bikes=loadImages(imFiles,(28,28))

    imFiles=[]
    for d in os.walk([dataDir,'planes/'):
        for f in d[2]:
            if ".directory" not in f:
                imFiles.append(d[0]+"/"+f)
    planes=loadImages(imFiles,(28,28))

