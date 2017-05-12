import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, generic_utils

def gen_Model(num_units, actfn='linear', reg_coeff=0.0, last_act='softmax'):
    ''' Generate a neural network model of approporiate architecture
    Args:
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Output:
        model: Keras sequential model with appropriate fully-connected architecture
    '''
    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=actfn,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=last_act,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model
def model_Train(X_tr, Y_tr, arch, actfn='sigmoid', last_act='sigmoid', reg_coeff=0.0,
                num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decay=0.0, sgd_mom=0.0,
                    sgd_Nesterov=False, EStop=False):
    call_ES = EarlyStopping(monitor='val_acc', patience=6, mode='auto')
    model = gen_Model(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, last_act=last_act)
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if EStop:
        model.fit(X_tr, Y_tr, nb_epoch=num_epoch, batch_size=batch_size, callbacks=[call_ES],
                   validation_split=0.1, validation_data=None, shuffle=True)
    else:
        model.fit(X_tr, Y_tr, batch_size=100, nb_epoch=10, shuffle=True, verbose=1,show_accuracy=True,validation_split=0.2)
    return model

def model_Pred(X_pred, model, batch_size = 1000):
    Y_pred = model.predict(X_pred, batch_size)
    return Y_pred






from imblearn.combine import SMOTEENN
sm = SMOTEENN()
train = np.loadtxt('train.txt')
test_temp = np.loadtxt('fit.txt')

word_train = np.loadtxt('word_train.txt')
word_test = np.loadtxt('word_test.txt')
word_final = np.loadtxt('word_final.txt')

zan_interest_train = np.loadtxt('zan_interest_train.txt')
zan_interest_test = np.loadtxt('zan_interest_test.txt')
zan_interest_final = np.loadtxt('zan_interest_final.txt')


final = np.loadtxt('test.txt')
def normal(data):
    data = (data - np.mean(data))/np.std(data)
    return data

poss_train = train[:,1]
poss_test = test_temp[:,1]
poss_final = final[:,1]

poss_word_train = poss_train * word_train
poss_word_test = poss_test * word_test
poss_word_final = poss_final * word_final

match_label_train = train[:,0]
match_label_test = test_temp[:,0]
match_label_final = final[:,0]

poss_label_train = train[:,2]
poss_label_test = test_temp[:,2]
poss_label_final = final[:,2]


zan_interest_train_match = zan_interest_train* match_label_train
zan_interest_test_match = zan_interest_test* match_label_test
zan_interest_final_match = zan_interest_final* match_label_final








X_train = np.vstack([match_label_train,word_train,poss_train,poss_label_train, zan_interest_train]).T

X_test = np.vstack([match_label_test,word_test,poss_test,poss_label_test, zan_interest_test]).T

X_final = np.vstack([match_label_final,word_final,poss_final,poss_label_final, zan_interest_final]).T


target =np.loadtxt('target.txt')




X_resampled, y_resampled = sm.fit_sample(X_train, target)

print 3
y_resampled = np_utils.to_categorical(y_resampled, 2)
model1 = model_Train(X_resampled,y_resampled, [5,50,50,50,2])
y_pred = model_Pred(X_test,model1)
pred = np.array(y_pred[:,1]).tolist()
np.savetxt('pred_validation.txt', pred)
y_final = model_Pred(X_final,model1)
pred_final = np.array(y_final[:,1]).tolist()
np.savetxt('pred_final.txt', pred_final)
