import numpy as np
from math import sqrt

class Expert:
    def __init__(self, data_line):
        self.id = data_line[0]
        self.label = data_line[1].split('/')
        if data_line[2] == '/':
            self.wordDesc = []  
        else:
            self.wordDesc = data_line[2].split('/')
        if data_line[3].strip() == '/':
            self.charaDesc = []
        else:
            self.charaDesc = data_line[3].strip().split('/')

class Question:
    def __init__(self, data_line):
        self.id = data_line[0]
        self.label = data_line[1]
        if data_line[2] == '/':
            self.wordDesc = []
        else:
            self.wordDesc = data_line[2].split('/')
        if data_line[3] == '/':
            self.charaDesc = []
        else:
            self.charaDesc = data_line[3].split('/')
        self.zan = int(data_line[4])
        self.ans = int(data_line[5])
        self.jing = int(data_line[6].strip())



def load(filename):
    with open(filename) as f:
        line = f.readlines()
    data = []
    for i in range(0, len(line)):
        data.append(line[i].split('\t'))
    return data

def load_test(filename):
    with open(filename) as f:
        line = f.readlines()
    data = []
    for i in range(1, len(line)):
        data.append(line[i].strip().split(','))
    return data

user_info = load('user_info.txt')
ques_info = load('question_info.txt')
invi_info_train = load('invited_info_train.txt')
vali = load_test('validate_nolabel.txt')
final = load_test('test_nolabel.txt')

e = {}
q = {}
for i in user_info:
    e[i[0]] = Expert(i)
for i in ques_info:
    q[i[0]] = Question(i) 

q_AnsBy_e = {}
for i in invi_info_train:
    if q_AnsBy_e.has_key(i[0]):
        q_AnsBy_e[i[0]].append([i[1], int(i[2])])
    else:
        q_AnsBy_e[i[0]] = [[i[1], int(i[2])]]


e_Ans_q = {}
for i in invi_info_train:
    if e_Ans_q.has_key(i[1]):
        e_Ans_q[i[1]].append([i[0], q[i[0]].label, int(i[2])])
    else:
        e_Ans_q[i[1]] = [[i[0], q[i[0]].label, int(i[2])]]

        
e_ind = {}
q_ind = {}
count = 0
for i in user_info:
    e_ind[i[0]] = [Expert(i), count]
    count += 1
count = 0
for i in ques_info:
    q_ind[i[0]] = [Question(i),count]
    count += 1
    
answer_or_not = np.zeros([len(e_ind), len(q_ind)])
user_dim = 0
for answer in q_AnsBy_e:
    for user_answering in q_AnsBy_e[answer]:
        user_axis = (e_ind[user_answering[0]])[1]
        ques_axis = (q_ind[answer])[1]
        answer_or_not[user_axis][ques_axis] = user_answering[1]
user_answer_num = np.sum(answer_or_not, axis = 1, dtype = float)
total_anser_num = np.sum(user_answer_num)

prob = [[0 for a in range(20)] for b in range(len(user_info))]
sent = [[0 for a in range(20)] for b in range(len(user_info))]
for i in range(len(user_info)):
    label = {}
    label_answered = {}
    if user_info[i][0] in e_Ans_q:
        for j in range(len(e_Ans_q[user_info[i][0]])):
            if e_Ans_q[user_info[i][0]][j][1] not in label:
                sent[i][int(e_Ans_q[user_info[i][0]][j][1])] = 1
                label[e_Ans_q[user_info[i][0]][j][1]] = 1
                label_answered[e_Ans_q[user_info[i][0]][j][1]] = 0
                if e_Ans_q[user_info[i][0]][j][2] == 1:
                    label_answered [e_Ans_q[user_info[i][0]][j][1]] = label_answered[e_Ans_q[user_info[i][0]][j][1]] + 1
            else:
                sent[i][int(e_Ans_q[user_info[i][0]][j][1])] = sent[i][int(e_Ans_q[user_info[i][0]][j][1])] + 1
                label[e_Ans_q[user_info[i][0]][j][1]] = label[e_Ans_q[user_info[i][0]][j][1]] + 1
                if e_Ans_q[user_info[i][0]][j][2] == 1:
                    label_answered [e_Ans_q[user_info[i][0]][j][1]] = label_answered[e_Ans_q[user_info[i][0]][j][1]] + 1
        prob[i][int(e_Ans_q[user_info[i][0]][j][1])] = float(label_answered[e_Ans_q[user_info[i][0]][j][1]])/float(label[e_Ans_q[user_info[i][0]][j][1]])
sent = np.array(sent)
recommend = np.sum(sent, axis = 1)

total_rec = np.sum(recommend)
poss = []       
for i in range(len(recommend)):
    if recommend[i] != 0:
        poss.append(user_answer_num[i]/recommend[i])
    else:
        poss.append(0)
poss = np.array(poss)
mean_poss = float(total_anser_num) / float(total_rec)
poss += mean_poss

num_of_q_label = 0
label_question = []      
for question in q:
    if q[question].label in label_question:
        pass
    else:
        num_of_q_label += 1
        label_question.append(q[question].label)


num_of_e_label = 0
label_user = []      
for expert in e:
    for label in e[expert].label:
        if label in label_user:
            pass
        else:
            num_of_e_label += 1
            label_user.append(label)

    

u_label_matrix = []
for expert in e:
    this_label = [0] * num_of_e_label
    for label in e[expert].label:
        this_label[int(label)] = 1
    u_label_matrix.append(this_label)
u_label_matrix = u_label_matrix - np.mean(u_label_matrix, axis = 0) / np.std(u_label_matrix, axis = 0)
u_label_matrix = np.mat(u_label_matrix, dtype = float)
u_label_cov = np.cov(u_label_matrix.T)
e_value, u_label_coeff = np.linalg.eig(u_label_cov)
u_label_coeff = np.mat(u_label_coeff)
u_label_coeff = u_label_coeff.I

ques_label_to_user_label = np.zeros([num_of_q_label, num_of_e_label])
for q_id in q_AnsBy_e:
    q_label = int(q[q_id].label)
    for u_info in q_AnsBy_e[q_id]:
        u_id = u_info[0]
        u_label = (e[u_id]).label
        u_label = np.array(u_label, dtype = int)
        u_has_label = np.zeros([num_of_e_label])
        u_has_label[u_label] = 1
        ques_label_to_user_label[q_label] += u_has_label

ques_label_to_user_label = np.mat(ques_label_to_user_label, dtype = float)
ques_label_to_user_label = ques_label_to_user_label.T
q_label_coeff = u_label_coeff * ques_label_to_user_label
q_label_coeff = np.array(q_label_coeff)

q_label_coeff = q_label_coeff.T
for coeff in q_label_coeff:
    norm = np.sum(np.square(coeff))
    if norm != 0:
        coeff /= sqrt(norm)
        
def user_Coeff(user_id):
    user_has_label = np.zeros([num_of_e_label])
    user_label = (e[user_id]).label
    user_label = np.array(user_label, dtype = int)
    user_has_label[user_label] = 1
    user_has_label = np.mat(user_has_label)
    user_has_label = user_has_label.T
    user_coeff = u_label_coeff * user_has_label
    user_coeff = np.array(user_coeff)
    norm = np.sum(np.square(user_coeff))
    if norm != 0:
        user_coeff /= sqrt(norm)
    return user_coeff

coeff_dict = {}

def user_Label_Match(user_id, ques_id, user_coeff_dict):
    user_coeff = np.array([])
    if user_coeff_dict.has_key(user_id):
        user_coeff = user_coeff_dict[user_id]
    else:
        user_coeff = user_Coeff(user_id)
        user_coeff_dict[user_id] = user_coeff
    ques_label = int(q[ques_id].label)
    ques_coeff = q_label_coeff[ques_label]
    match = np.dot(ques_coeff, user_coeff)
    match = match[0]
    return match, user_coeff_dict

def input_Calculation(dataset, u_coeff_dict):
    u_prob_to_q = []
    for this_test in dataset:
        q_id_data = this_test[0]
        u_id_data = this_test[1]
        label_match, u_coeff_dict = user_Label_Match(u_id_data, q_id_data, u_coeff_dict)
        u_index = (e_ind[u_id_data])[1]
        answer_poss = poss[u_index]
        u_prob_to_q.append([label_match, answer_poss, label_match * answer_poss])
    return u_prob_to_q, u_coeff_dict

input_train, coeff_dictf = input_Calculation(invi_info_train, coeff_dict)    
input_vali, coeff_dictf = input_Calculation(vali, coeff_dictf)    
input_test, coeff_dictf = input_Calculation(final, coeff_dictf)

np.savetxt('train.txt', input_train)
np.savetxt('fit.txt', input_vali)
np.savetxt('test.txt', input_test)

ans = 0
for i in invi_info_train:
    ans += int(i[2])
total_ans_mean =  float(ans) / float(len(invi_info_train))

zan_level = {}
for i in q:
    if q[i].zan < 13:
        zan_level[i] = 0
    elif q[i].zan < 57:
        zan_level[i] = 1
    elif q[i].zan < 193:
        zan_level[i] = 2
    elif q[i].zan < 720:
        zan_level[i] = 3
    else:
        zan_level[i] = 4

def getZanLev(level, u_index):
    rec = 0
    ans = 0
    t_rec = 0
    t_ans = 0
    if not e_Ans_q.has_key(user_info[u_index][0]):
        return total_ans_mean
    for i in e_Ans_q[user_info[u_index][0]]:
        t_rec += 1
        if int(i[1]) == 1:
            t_ans += 1
        if zan_level[i[0]] == level:
            rec += 1
            if int(i[1]) == 1:
                ans += 1
    if rec == 0:
        return (float(t_ans) / float(t_rec))
    return (float(ans) / float(rec))

zan_matrix = np.zeros([5, len(e)])
for i in range(0, 5):
    for j in range(0, len(e)):
        zan_matrix[i][j] = getZanLev(i, j)


zan_train = []              
for q_to_e in q_AnsBy_e:
    for expert_expected in q_AnsBy_e[q_to_e]:
        zan_level_this = zan_level[q_to_e]
        id_u = expert_expected[0]
        ind_u = (e_ind[id_u])[1]
        zan_train.append(zan_matrix[zan_level_this][ind_u])

zan_test = []
for q_to_e in vali:
    id_q = q_to_e[0]
    id_u = q_to_e[1]
    zan_level_this = zan_level[id_q]
    ind_u = (e_ind[id_u])[1]
    zan_test.append(zan_matrix[zan_level_this][ind_u])
    
zan_final = []
for q_to_e in final:
    id_q = q_to_e[0]
    id_u = q_to_e[1]
    zan_level_this = zan_level[id_q]
    ind_u = (e_ind[id_u])[1]
    zan_final.append(zan_matrix[zan_level_this][ind_u])

np.savetxt('zan_interest_train.txt', zan_train)
np.savetxt('zan_interest_test.txt', zan_test)
np.savetxt('zan_interest_final.txt', zan_final)

def word(test, i):
    temp_word = 0.0
    for j in e[test[i][1]].wordDesc:
        if j in q[test[i][0]].wordDesc:
            temp_word = temp_word + 1.0
    total_user = len(e[test[i][1]].wordDesc)
    total_word = len(q[test[i][0]].wordDesc)
    if total_user == 0 or total_word == 0:
        return 0
    return    temp_word*temp_word/total_user/total_word

def word_Look(test):
    word_look = []
    for i in range(len(test)):
        word_look.append(word(test, i))
    return word_look

word_look_train = word_Look(invi_info_train)
word_look_vali = word_Look(vali)
word_look_test = word_Look(final)

np.savetxt('word_train.txt', word_look_train)
np.savetxt('word_test.txt', word_look_vali)
np.savetxt('word_final.txt', word_look_test)
target = []
for i in invi_info_train:
    target.append(int(i[2]))

np.savetxt('target.txt', target)