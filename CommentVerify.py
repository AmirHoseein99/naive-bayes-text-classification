import csv
import numpy as np

stopword = set()
valsens = []
invalsens = []
classWords = {0: 0,
              1: 0}
WordsVClass = {"پاور": 0, }
WordsInVClass = {}
logprior = {}
inVLogLikeHood = {}
vLogLikeHood = {}
allclasse = set()
allclasse.add(0)
allclasse.add(1)
classSents = {}
allWords = set()


def preprocess():
    with open('/home/amirhossein/AI-comment/StopWords/chars', 'r') as stop:
        for line in stop:
            stopword.add(line)

    with open('/home/amirhossein/AI-comment/train/train.csv', 'r') as f:
        train = csv.reader(f, delimiter=',')
        for row in train:
            if row[4] == "0":
                valsens.append(row[1])
                valsens.append(row[2])
            if row[4] == "1":
                invalsens.append(row[1])
                invalsens.append(row[2])
            for word in row[1].split(" "):
                if word in stopword:
                    pass
                else:
                    allWords.add(word)
                if row[4] == '1':
                    classWords[1] += 1
                if row[4] == "0":
                    classWords[0] += 1
            for word in row[2].split(" "):
                if word in stopword:
                    pass
                else:
                    allWords.add(word)
                if row[4] == '1':
                    classWords[1] += 1
                if row[4] == "0":
                    classWords[0] += 1

    classSents[0] = valsens
    classSents[1] = invalsens


def learning():
    for sent in classSents[0]:
        valWords = sent.split(" ")
        for word in valWords:
            if word in WordsVClass:
                WordsVClass[word] += 1
            else:
                WordsVClass[word] = 1
    for sent in classSents[1]:
        invalWords = sent.split(" ")
        for word in invalWords:
            if word in WordsInVClass:
                WordsInVClass[word] += 1
            else:
                WordsInVClass[word] = 1

    logprior[0] = np.log((len(valsens))/(len(valsens)+len(invalsens)))
    logprior[1] = np.log(((len(invalsens))/(len(valsens)+len(invalsens))))

    for word in allWords:
        if word in WordsVClass:
            valcount = WordsVClass[word]
        if word in WordsInVClass:
            inValcount = WordsInVClass[word]
        vLogLikeHood[word] = np.log(
            (valcount+1) / ((classWords[0] + 1) * len(allWords)))
        inVLogLikeHood[word] = np.log(
            (inValcount+1) / ((classWords[1] + 1) * len(allWords)))


def predict():
    with open('/home/amirhossein/AI-comment/test/test.csv', 'r') as testFile:
        with open('/home/amirhossein/AI-comment/Ans/ans.csv', 'w') as resualt:
            resultWriter = csv.writer(resualt)
            test = csv.reader(testFile, delimiter=',')
            resultWriter.writerow(['id', 'verification_status'])
            sums = {
                0: 0,
                1: 0,
            }
            for row in test:
                sums[0] = logprior[0]
                sums[1] = logprior[1]
                titleallWords = row[1].split(" ")
                commentallWords = row[2].split(" ")
                for word in titleallWords:
                    if word in allWords:
                        sums[1] += inVLogLikeHood[word]
                        sums[0] += vLogLikeHood[word]
                for word in commentallWords:
                    if word in allWords:
                        sums[1] += inVLogLikeHood[word]
                        sums[0] += vLogLikeHood[word]
                if sums[0] > sums[1]:
                    resultWriter.writerow([f'{row[0]}', '0'])
                else:
                    resultWriter.writerow([f'{row[0]}', '1'])


def main():
    preprocess()
    learning()
    predict()


if __name__ == '__main__':
    main()
