import numpy as np
import re
import random as rd
def sparse_text(text_string):
    split_string = re.split("\\W*", text_string)
    return [word.lower() for word in split_string if len(word) > 2]

def load_data():
    email_text_list = []
    label_list = []
    for i in range(1, 26):
        email_text = sparse_text(open("email/spam/%d.txt" %i, encoding = "ISO-8859-1").read())
        email_text_list.append(email_text)
        label_list.append(1)
        email_text = sparse_text(open("email/ham/%d.txt" %i, encoding = "ISO-8859-1").read())
        email_text_list.append(email_text)
        label_list.append(0)   
    return (email_text_list, label_list)

def create_token_list(data_list):
    token_set = set([])
    for data in data_list:
        token_set |= set(data)
    return list(token_set)
    

def create_token_matrix(token_list, comment_list):
    token_matrix = []
    for a_comment in  comment_list: 
        data_list = [0] * len(token_list)
        for token in a_comment:
            if token in token_list:
                data_list[token_list.index(token)] = 1
        token_matrix.append(data_list)
    return token_matrix

def calc_prob(token_list, label_list):
    token_matrix = np.array(token_list)
    label_num = len(set(label_list))
    instance_len = token_matrix.shape[1]
    instance_num = token_matrix.shape[0]
    w_list = np.ones([label_num, instance_len])
    prob_vector = np.zeros([label_num, instance_len])
    w_num = np.zeros(label_num) + instance_len
    labels = list(set(label_list))
    prior_prob = np.zeros(label_num)
    prior_num = np.zeros(label_num)
    
    for i in range(instance_num):
        for j in range(label_num):
            if label_list[i] == labels[j]:
                prior_num[j] += 1
                w_list[j] += token_matrix[i]
                w_num[j] += sum(token_matrix[i])
    
    prior_prob = prior_num / float(len(label_list))
    prob_vector = np.log(w_list / np.tile(w_num, instance_len).reshape((instance_len, label_num)).transpose(1,0))
    return (prob_vector, prior_prob, labels)

def classify(predict_instance, prob_vector, prior_prob, labels):
    prob = (predict_instance * prob_vector).sum(axis = 1) + prior_prob
    index = np.where(prob == max(prob))[0][0]
    return labels[index]


def bayes_classify(predict_instance):
    email_text_list, label_list = load_data()
    token_list = create_token_list(email_text_list)
    token_matrix = create_token_matrix(token_list, email_text_list)
    prob_vector, prior_prob, labels = calc_prob(token_matrix, label_list)
    predict_array = np.array(create_token_matrix(token_list, [predict_instance]))
    final_class = classify(predict_array, prob_vector, prior_prob, labels)
    return final_class

def test():
    email_text_list, label_list = load_data()
    error_count = 0
    for j in range(10):
        training_set = [i for i in range(50)]
        test_set = rd.sample(training_set, 10)
        for i in test_set:
            del(training_set[training_set.index(i)])
     
        training_matrix = []
        training_class = []
        for i in training_set:
            training_matrix.append(email_text_list[i])
            training_class.append(label_list[i])
        token_list = create_token_list(training_matrix)
        token_matrix = create_token_matrix(token_list, training_matrix)
        prob_vector, prior_prob, labels = calc_prob(token_matrix, training_class)
        for i in test_set:
            token_instance = np.array(create_token_matrix(token_list, [email_text_list[i]]))
            if classify(token_instance, prob_vector, prior_prob, labels) != label_list[i]:
                error_count += 1
    print("The error rate is: ", float(error_count) / len(test_set) / 10.0)
      
if __name__ == "__main__":
    test()
