# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from numpy import *
import time

train_spam_count = 0
train_ham_count = 0
test_spam_count = 0
test_ham_count = 0
need_study = False
# 创建一些样本raw_data_list，以及对应的样本类别label，这些样本（raw_data_list）是训练样本，
# 之后的贝叶斯分类器，就依据这些数据来分类你输入的数据的。
def load_data_set(data_path):
    # raw_data_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
    #              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    #              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    #              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    #              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    #              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    '''
    label中的0代表着褒义， 1代表着贬义。label数组中一共有6个元素，分别代表着
    raw_data_list"每一行"的数组所对应的语义是褒是贬。
    比如"第一行"：
    ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
    对应label的第一个元素：0，也就是说第一行是褒义的，
    而"第二行"：
    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
    则对应label的第二个元素：1，也就是说第二行是贬义的。
    '''
    # label = [0, 1, 0, 1, 0, 1]
    print('Start read data')
    before_read = time.time()
    raw_data = pd.read_csv(data_path, header=0)
    end_read = time.time()
    print('Finished read data')
    print('Reading email costs %ss' % (end_read - before_read))
    data = raw_data.values
    emails = data[:2000, 1::]
    label = data[:2000, 0]
    label = list(map(lambda x: 0 if x == 'ham' else 1, label))

    test_emails = data[2000:2500, 1::]
    test_label = data[2000:2500, 0]
    test_label = list(map(lambda x: 0 if x == 'ham' else 1, test_label))
    return emails, label, test_emails, test_label


'''
    将方法load_data_set执行后返回的raw_data_list，处理成没有重复元素的列表
    至于为何要将raw_data_list转换成列表，大概是因为raw_data_list是一个二维数组，
    每个数组元素又是一个一维数组，而且这个一维数组并不知道具体长度，结构不一致，
    不方便之后的计算，而转换成列表后，计算就方便多了。
'''
def create_no_repeat_data_list(data):
    # 创建一个空的set集合，python里的set是不会出现重复的元素的，
    # 使用set经常用于数据去重。
    data_set = []
    index = 0
    for item in data:
        print('read %s email into set' % (index + 1))
        item = str(item[0]).split(',')
        data_set.extend(item)
        index += 1

    # 转换成list，并返回。
    # print list(data_set)
    return list(set(data_set))

'''
    参数data_list可以的是方法create_no_repeat_data_list执行后，所返回的list，
    参数input_word，则就是你所输入的话，
    这个方法的作用就是标记data_set中的某一条训练样本所出现的单词，在训练样本data_list里的次数，
    若是出现了则标记为1，没有则是0
    PS：data_list是一个列表，将二位数组data_set去重后，将其所有的元素都放进了data_list中。
    detect_word(data_list, data)
'''
def detect_word(data_list, input_word):
    # 创建和训练样本一样大的数组，该数组每个元素都是0，因为还没开始标记嘛。
    return_vec = [0]*len(data_list)
    # print input_word

    for word in input_word:
        if word in data_list:
            # 将出现在样本里的单词，在对应的位置上的数量加1。
            # if word == 'saw':
                # print '^'*20
                # print data_list.index(word)
            return_vec[data_list.index(word)] += 1
        else:
            need_study = True
            # print "The word :%s is not in the vocabulary!" % word

    # print return_vec
    # 将统计好的结果返回，这个结果之后会用到，用于概率计算。
    return return_vec

'''
    参数train_martix：可以是 方法detect_word执行后，所产生的数组，例如：
    trainMatrix = []
    for data in data_set:
        trainMatrix.append(detect_word(data_list, data))

    这个数组记录了每条训练样本里所出现的单词，在整个训练样本中，所出现的次数。
    参数train_category：指的是每个训练样本所对应的类别。

    这个方法的作用是根据train_martix这个记录次数的数组来计算，对应训练样本的类别的情况下，
    是这个单词的概率。比如当该条训练样本是贬义的情况下，当前单词是'dog'这个单词的概率。
'''
def train_nb0(train_martix, train_category):
    global train_ham_count
    global train_spam_count
    # 训练样本的总量。
    total_num = len(train_martix)
    # 获取整个样本里有多少个单词。
    words_vec_length = len(train_martix[0])
    # train_category，比如[0, 1, 1, 0], 0代表褒义， 1代表贬义，将其求和，就是贬义的样本的总数：0+1+1+0=2。
    # 贬义的样本的总数/样本总数=在整个样本中，随机一个样本是贬义样本的概率。
    p_bad_word = sum(train_category) / float(total_num)

    # 初始化一个数组，这个数组用来记录，当该训练样本是贬义时，整个样本中每个单词所出现的次数
    # zeros这个方法是创一个每个元素全为0的，长度为words_vec_length的数组
    # bad_vec = zeros(words_vec_length)
    # 但我们没用zeros来初始化数组，而改用了ones，即每个元素都为1，原因是因为，有些数据确实没有出现过的次数，
    # 这时候他就是0，概率在之后的计算中是需要相乘的，0乘谁都是0，概率会过于的小，反而不准确了。
    bad_vec = ones(words_vec_length)
    # 初始化一个数组，这个数组用来记录，当该训练样本是褒义时，整个样本中每个单词所出现的次数
    # good_vec = zeros(words_vec_length)
    good_vec = ones(words_vec_length)
    # 用于记录，当该训练样本是贬义时，整个样本中所出现的单词的总数
    # bad_word_total_num = 0.0
    # 弃用初始值是0的原因是因为bad_vec的初始值都是1，之后的计算需要bad_vec/bad_word_total_num，
    # 分子初始值变大了，分母的初始值也需要变大
    bad_word_total_num = 2.0
    # 用于记录，当该训练样本是褒义时，整个样本中所出现的单词的总数
    # good_word_total_num = 0.0
    good_word_total_num = 2.0

    # 遍历训练样本
    for index in range(total_num):
        # print 'train %s email...' % (index)
        start_time = time.time()
        # 如果这个样本是贬义的
        if train_category[index] == 1:
            train_spam_count += 1
            # train_martix是记录这条训练样本里所出现的单词次数的数组，比如
            # bad_vec是[0,0,0,0,0], train_martix[index]是[0,0,1,0,1],
            # train_martix可以是[[0,0,1,0,1],
            #                   [0,0,0,0,1]
            #                   [0,0,1,0,0]
            #                   [1,0,0,0,0]
            #                   [1,1,1,0,1]
            #                   [1,0,0,0,1]
            #                   ]
            bad_vec += train_martix[index]
            # sum(train_martix[index])就是这个训练样本里一共出现了几个单词，
            # 比如[0,0,1,0,1]，求和后就是2，一共出现了2个单词
            bad_word_total_num += sum(train_martix[index])
        else:
            train_ham_count += 1
            # 同理如上
            good_vec += train_martix[index]
            good_word_total_num += sum(train_martix[index])
        end_time = time.time()
        # print 'cost time: %s' % (end_time - start_time)

    # bad_vec用来记录，当该训练样本是贬义时，整个样本中每个单词所出现的次数，
    # bad_word_total_num用来记录，训练样本是贬义时，整个样本中所出现的单词的总数
    # 两个相除，所得出的数组p_bad_vec，记录着当样本为贬义时，每个单词所出现的概率。
    p_bad_vec = bad_vec / bad_word_total_num
    # 同上，结果是当样本为褒义时，每个单词所出现的概率。
    p_good_vex = good_vec / good_word_total_num

    # print "bad_word_total_num, good_word_total_num"
    # print bad_word_total_num, good_word_total_num

    # 之所以返回log数，是因为计算出来的概率都特别的小，采用log后，数值会增大，
    # 而且log函数的变化趋势并不影响原先的概率
    # print log(p_bad_vec), log(p_good_vex), p_bad_word
    # print '&'*10
    # print p_bad_vec[499]
    # print p_good_vex[499]
    # print bad_vec[499]
    # print good_vec[499]
    # print log(p_bad_vec[499])
    # print log(p_good_vex[499])
    return log(p_bad_vec), log(p_good_vex), p_bad_word

'''
    这个方法的作用：
    你所输入的话，对比其是褒义的概率，以及是贬义的概率，
    哪个概率大，就是哪一种话。

    参数：vec_array，他是一个数组，记录你所说的话，在样本中所出现的次数，比如：
    test_array = ['I', 'love', 'python']，test_array就是你所说的话。
    vec_array = array(detect_word(data_list, []))

    参数p_bad_vec就是train_nb0执行后所得出的：当样本为贬义时，每个单词所出现的概率。
    参数p_good_vex同上，是当样本为褒义时，每个单词所出现的概率。
    参数p_bad_word同上，是随机一个样本是贬义的概率。
'''
def classify_nb(vec_array, p_bad_vec, p_good_vex, p_bad_word):
    # vec_array*p_bad_vec：vec_array比如[1,0,0],乘p_bad_vec后，得到的就是，话是贬义时，
    # 话里的每个单词，那些在样本中存在的单词，他们出现的概率。
    # 将这个单词出现的概率相乘就是，当话是贬义时，正好是你这句话的概率，将这个概率记为P(A|B),
    # P(B)代表着话是贬义的概率。P(AB) = P(A|B) * P(B)，P(AB)就是你这句话是贬义的概率。

    # 因为p_bad_vec已经经过log处理过了，所以p_bad_vec里面所有元素的相乘等于相加，
    # log(a*b*c) = log(a) + log(b) +log(c)
    # sum(vec_array*p_bad_vec)得到的就是P(A|B)，再加上log(p_bad_word)，相当于乘P(B)
    # sum(vec_array*p_bad_vec) + log(p_bad_word)便得到了这句话是贬义的概率
    p1 = sum(vec_array*p_bad_vec) + log(p_bad_word)
    # 同上，这句话是褒义的概率。
    p2 = sum(vec_array*p_good_vex) + log(1 - p_bad_word)
    # for i in (vec_array*p_bad_vec):
        # print(i)
    # print '*'*20
    # for i in vec_array*p_good_vex:
        # print(i)
    # print(p_bad_word)

    # print '*'*20
    # print p1,p2

    if p1 >= p2:
        return 1
    else:
        return 0

def classifiy(words, p_bad_vec, p_good_vex, p_bad_word):
    vec_array = array(detect_word(data_list, words))
    return classify_nb(vec_array, p_bad_vec, p_good_vex, p_bad_word)



if __name__ == '__main__':
    global test_ham_count
    global test_spam_count
    data_set, label, test_emails, test_label = load_data_set('email_data.csv')
    data_list = create_no_repeat_data_list(data_set)
    # i = 0
    # for item in data_list:
        # if item == 'saw':
            # pass
            # print '-'*20
            # print i
        # i += 1
    # print '总词向量数：%s' % (len(data_list),)
    # print data_list

    p_start_time = time.time()
    trainMatrix = []
    index = 0
    for data in data_set:
        # start_time = time.time()
        # print 'calculate train matrix with %s email' % (index+1,)
        data = str(data[0]).split(',')
        trainMatrix.append(detect_word(data_list, data))
        # end_time = time.time()
        # print 'cost time:%s' % (end_time - start_time)
        index += 1
    #p_start_time = time.time()
    p_bad_vec, p_good_vex, p_bad_word = train_nb0(trainMatrix, label)
    after_time = time.time()
    print('train cost:%ss' % (after_time - p_start_time))

    index = 0
    I, J, K, L = (0, 0, 0, 0)
    p_start_time = time.time()
    for test_email in test_emails:
        words = str(test_email[0]).split(',')
        # print 'classifiy %s email' % (index,)
        result = classifiy(words, p_bad_vec, p_good_vex, p_bad_word)
        correct_label = test_label[index]
        # correct_label 0 -> ham
        if correct_label == 0:
            test_ham_count += 1
            if result == correct_label:
                I += 1
            else:
                J += 1
        else:
            test_spam_count += 1
            if result == correct_label:
                L += 1
            else:
                K += 1
        index += 1

    print('I: %s, J: %s, K: %s, L: %s' % (I, J, K, L))
    print('correct rate: %s' % ((float(L + I) / (I + J + K + L))))
    print('ham email correct rate: %s' % (float(I) / (K + I)))
    print('spam email correct rate: %s' % (float(L) / (L + J)))
    print('ham email return rate: %s' % (float(I) / (I + J)))
    print('spam email return rate: %s' % (float(L) / (L + K)))

    print('train_spam_count: %s, train_ham_count: %s, test_spam_count: %s, test_ham_count: %s' % (
    train_spam_count, train_ham_count, test_spam_count, test_ham_count))









