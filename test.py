from sklearn.ensemble import RandomForestClassifier

domainlist = []
testDomainlist = []
class Domain:
    def __init__(self, _name, _label, _length, _entropy):
        self.name = _name
        self.label = _label
        self.length = _length
        self.entropy = _entropy

    def returnData(self):
        return [self.length, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return "dga"
        else:
            return "nodga"

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(tokens[0]) 
            entropy = Entropy(tokens[0])
            domainlist.append(Domain(name,label,length,entropy))

class Testdomain:
    def __init__(self, _name, _length, _entropy):
        self.name = _name
        self.length = _length
        self.entropy = _entropy

    def returnTestdata(self):
        return [self.length, self.entropy]

def initTestdata(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]            
            length = len(tokens[0]) 
            entropy = Entropy(tokens[0])
            testDomainlist.append(Testdomain(name,length,entropy))

def Entropy(domain):
    length = len(domain)
    amount = 0
    for letter in domain:
        if 48 <= ord(letter) <= 57:
            amount = amount+1
    entropyOfdomain = amount/length
    return entropyOfdomain

initData("train.txt")
featureMatrix = []
labelList = []
for item in domainlist:
    featureMatrix.append(item.returnData())
    labelList.append(item.returnLabel())

clf = RandomForestClassifier(random_state=0)
clf.fit(featureMatrix,labelList)

initTestdata("test.txt")
testFeatureMatrix = []
testLabelList = []
for item in testDomainlist:
    testFeatureMatrix.append(item.returnTestdata())

resultList = clf.predict(testFeatureMatrix)

fresult = open("result.txt","w")
for i in range(len(resultList)):
    fresult.write(testDomainlist[i].name)
    fresult.write(",")
    fresult.write(resultList[i])
    fresult.write("\n")
fresult.close()
