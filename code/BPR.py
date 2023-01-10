
import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores
from sklearn.metrics import accuracy_score

class BPR:
    user_count = 3923#3923
    item_count = 3057#3057
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    train_count =20000
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01
    V = np.random.rand(item_count, latent_factors) * 0.01
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)#把2-d的矩阵平铺开了
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user -1][item-1 ] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)[0]
            # sample a negative item from the unobserved items
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            u -= 1
            i -= 1
            j -= 1
            r_ui = np.dot(self.U[u], self.V[i].T)#分解成了两个矩阵
            r_uj = np.dot(self.U[u], self.V[j].T)
            r_uij = r_ui - r_uj#做矩阵减肥
            mid = 1.0 / (1 + np.exp(r_uij))#求指数
            temp = self.U[u]
            self.U[u] += -self.lr * (-mid * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (-mid * temp + self.reg * self.V[i])
            self.V[j] += -self.lr * (-mid * (-temp) + self.reg * self.V[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)#读入train.txt
        self.load_test_data(self.test_data_path)#读入test.txt
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        for i in range(self.user_count * self.item_count):
            self.test[i] = int(self.test[i])
        # training
        for i in range(self.train_count):
            self.train(user_ratings_train)
        predict_matrix = self.predict(self.U, self.V)
        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        #for i in range(self.user_count * self.item_count):
            #self.predict_[i] = int(self.predict_[i])
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
####
        #a_s=accuracy_score(self.test, self.predict_)
        #print('accuracy_score',a_s)
        #TP=1#TP（True Positive）：预测为正，实际为正
        #TN=1#TN（True Negative）:预测为负，实际为负
        #FN=1#FN（False Negative）：预测为负，实际为正
        #FP=1#FP（False Positive）:预测为正，实际为负
        #cout=0
        #for i in range(self.user_count * self.item_count):
            #if self.predict_[i]==1 and self.test[i]==1:
                #TP+=1
            #elif self.predict_[i]==0 and self.test[i]==0:
                #TN+=1
            #elif self.predict_[i]==0 and self.test[i]==1:
                #FN+=1
            #elif self.predict_[i]==1 and self.test[i]==0:
                #FP+=1
            #if self.test[i]==1:
                #cout+=1
        #print('测试集有多少hit:',cout)
        #R = TP / (TP+FN)
        #P = TP / (TP+FP)
        #A=(TP+TN)/(TP+TN+FN+FP)
        #N=TN/(TN+FN)
        #print('召回率:',R)
        #print('精确率：',P)
        #print('准确率:',A)
        #print('负精率:',N)
        #print(TP)

            
            

####
        # Top-K evaluation
        str(scores.topK_scores(self.test, self.predict_, 20, self.user_count, self.item_count))

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u-1) * item_count + j - 1]= 0
    return predict

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()
