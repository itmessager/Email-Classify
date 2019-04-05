# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from numpy import *
import time
import math

train_spam_count = 0
train_ham_count = 0
test_spam_count = 0
test_ham_count = 0


def parse_email_content(emails):
    contents = emails[0::, 0]
    data_set = []

    for content in contents:
        try:
            data_set.append(str(content).split(','))
        except Exception as e:
            print('*' * 20)
            print(content)
            raise e

    return data_set


'''
    统计所有邮件中，垃圾邮件出现单词的数量，正常邮件出现的数量，
    每个单词在垃圾邮件中出现的数量以及每个单词在正常邮件中出现的数量。
'''


def statistical_words(data, labels):
    global train_ham_count
    global train_spam_count
    spam_dict = {}
    ham_dict = {}
    total_spam = 2
    total_ham = 2
    total_words = []
    total_spam_email = 0
    index = 0

    # items是每一行
    for items in data:
        print('runing... index:%s' % (index,))
        label = labels[index]
        temp_dict = ham_dict
        if label == 'spam':
            total_spam_email += 1
            train_spam_count += 1
            temp_dict = spam_dict
        else:
            train_ham_count += 1

        # item是每一行出现的单词
        for item in items:
            total_words.append(item)
            num = temp_dict.get(str(item), 1)
            num += 1
            temp_dict[str(item)] = num
            if label == 'spam':
                num = ham_dict.get(str(item), 1)
                ham_dict[str(item)] = num
                total_spam += 1
            else:
                num = spam_dict.get(str(item), 1)
                spam_dict[str(item)] = num
                total_ham += 1
        index += 1
    # print spam_dict
    # print ham_dict

    return spam_dict, ham_dict, total_spam, total_ham, float(total_spam_email) / index

def classify_by_words(words, spam_dict, ham_dict, total_spam, total_ham, p_spam_email):
    p_total_spam = 0
    p_total_ham = 0

    for word in words:
        # print 'current word:'
        # print word

        spam_num = spam_dict.get(str(word), None)
        ham_num = ham_dict.get(str(word), None)
        if not (spam_num and ham_num):
            continue

        # p_total_spam += log(math.sqrt(float(spam_num) / total_spam))
        # p_total_ham += log(math.sqrt(float(ham_num) / total_ham))

        p_total_spam += log(math.sqrt(math.sqrt(float(spam_num) / total_spam)))
        p_total_ham += log(math.sqrt(math.sqrt(float(ham_num) / total_ham)))

        # print float(spam_num) / total_spam
        # print float(ham_num) / total_ham
        # print log(float(spam_num) / total_spam), log(float(ham_num) / total_ham)
    p_spam = p_total_spam + log(p_spam_email)
    p_ham = p_total_ham + log(1 - p_spam_email)
    # print '%'*20
    # print p_spam, p_ham
    if p_spam > p_ham:
        return 'spam'
    else:
        return 'ham'


if __name__ == '__main__':
    global test_ham_count
    global test_spam_count
    print('Start read data')
    before_read = time.time()

    raw_data = pd.read_csv('email_data.csv', header=0)

    end_read = time.time()
    print('Finished read data')
    print('Reading email costs %ss' % (end_read - before_read))
    data = raw_data.values
    emails = data[:2000, 1::]
    label = data[:2000, 0]

    print('Start parse')
    data_set = parse_email_content(emails)
    print('Finished parse')

    print('Start statistical words ')
    before_start = time.time()
    spam_dict, ham_dict, total_spam, total_ham, p_spam_email = statistical_words(data_set, label)
    after_start = time.time()
    print('train cost:%ss' % (after_start - before_start))

    # words = raw_input('Please input words:\n')

    print('Calculate correct rate...')
    test_emails = data[2000:2500, 1::]
    test_label = data[2000:2500, 0]
    index = 0

    I, J, K, L = (0, 0, 0, 0)
    before_start = time.time()
    for test_email in test_emails:
        words = str(test_email[0]).split(',')
        result = classify_by_words(words, spam_dict, ham_dict, total_spam, total_ham, p_spam_email)
        correct_label = test_label[index]
        if correct_label == 'ham':
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

    after_start = time.time()
    print('Test Cost:%ss' % (after_start - before_start))

    print('train_spam_count: %s, train_ham_count: %s, test_spam_count: %s, test_ham_count: %s' % (
        train_spam_count, train_ham_count, test_spam_count, test_ham_count))
