import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter

def LoadData(filepath):
    ratings = pd.read_csv(filepath, header=0)
    ratings = ratings[['user_id','item_id']]

    train = []
    for idx, row in ratings.iterrows():
        user = int(row['user_id'])
        item = int(row['item_id'])
        train.append([user, item])
    return PreProcessData(train)

def PreProcessData(originData):
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData


class ItemCF(object):
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() 

    def similarity(self):
        N = defaultdict(int) 
        for user, items in self._trainData.items():
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        recommends = dict()
        items = self._trainData[user]
        for item in items:
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue 
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        #return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])
        dict1=dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])
        list1=[list(dict1)[0],list(dict1)[1]]
        f=open('output.txt','a+')
        f.write(str(list1)+'\n')
        f.close()
        return dict1

    def train(self):
        self.similarity()

if __name__ == "__main__":
    train = LoadData("data_train.csv")
    print("train data size: %d" % (len(train)))
    ItemCF = ItemCF(train, similarity='iuf', norm=True)
    ItemCF.train()
    for i in range(3924):
        ItemCF.recommend(i, 2, 80)

    