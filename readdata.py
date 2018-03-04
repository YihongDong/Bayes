def readdata(filepath,height,weight,feetsize,label,sex=1):
    fr = open(filepath)
    arrayOfLine = fr.readlines()
    for line in arrayOfLine:
        dataOfLine = line.strip().split()
        height.append(dataOfLine[0])
        weight.append(dataOfLine[1])
        feetsize.append(dataOfLine[2])
        label.append(sex)
