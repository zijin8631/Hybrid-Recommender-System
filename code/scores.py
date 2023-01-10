

import heapq
import numpy as np
import math

def topK_scores(test, predict, topk, user_count, item_count):
    PrecisionSum = np.zeros(topk+1)
    RecallSum = np.zeros(topk+1)
    F1Sum = np.zeros(topk+1)
    NDCGSum = np.zeros(topk+1)
    OneCallSum = np.zeros(topk+1)
    DCGbest = np.zeros(topk+1)
    dic11 = {}#建一个dic
    #每个都初始化为一个一维的list
    MRRSum = 0
    MAPSum = 0
    hit_sum = 0
    total_test_data_count = 0
    file = open('output.txt', 'r')
    list_other=[]
    for line in file:
        list_sub_other=[]
        line = line.split(' ')
        list_sub_other.append(int(line[0]))
        list_sub_other.append(int(line[1]))
        list_other.append(list_sub_other)
    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    for i in range(user_count):
        dic11[i]={}
        user_test = []
        user_predict = []
        test_data_size = 0
        for j in range(item_count):
            if test[i * item_count + j] == 1.0:
                test_data_size += 1#test为1
            user_test.append(test[i * item_count + j])
            user_predict.append(predict[i * item_count + j])
        if test_data_size == 0:
            continue
        else:
            total_test_data_count += 1#数值上等于user_count
        predict_max_num_index_list = map(user_predict.index, heapq.nlargest(topk, user_predict))#找除给该user推荐的最可能的20个
        predict_max_num_index_list = list(predict_max_num_index_list)#这20个的index存在predict_max_num_index_list里面
        predict_max_num_index_list=predict_max_num_index_list[0:2]
        #print(predict_max_num_index_list)
        list_other1=list_other[i+1]

        predict_max_num_index_list.append(list_other1[0]-1)#外部的txt输入的两个推荐物品
        predict_max_num_index_list.append(list_other1[1]-1)
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)
        for k in range(1, 5):
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1] #
            if user_test[item_id] == 1:
                hit_sum += 1#说明推荐对了，也就是TP
                DCG[k] += 1 / math.log(k + 1)

            # precision, recall, F1, 1-call
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            f1 = 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            F1Sum[k] += float(f1)
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]
            if hit_sum > 0:
                OneCallSum[k] += 1
            else:
                OneCallSum[k] += 0
        # MRR
        p = 1
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                break
            p += 1
        MRRSum += 1 / float(p)
        # MAP
        p = 1
        AP = 0.0
        hit_before = 0
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                AP += 1 / float(p) * (hit_before + 1)
                hit_before += 1
            p += 1
        MAPSum += AP / test_data_size
    sum_k=0
    print(len(test))
    print(hit_sum)
    print('average precision is:',hit_sum/(user_count)/4)


    return
