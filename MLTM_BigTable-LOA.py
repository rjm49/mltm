
# coding: utf-8

# # BigTable MTLM for LOA
# (Based on Cen 2009)
# This model is used to explain the power law in learning.  In this notebook we try to build a neuralised version of the AFM and train it using simulated data.  The aim of using the AFM is to disentangle the latent traits that make up the overall score going into the sigmoid probability estimator.
# 
# The model is compensatory, which is a weakness.

# In[1]:


from collections import defaultdict, Counter
from copy import copy
from math import exp, sqrt, log
from random import random, shuffle, choice, randint, uniform
import numpy
import math

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.constraints import non_neg, max_norm
from numpy import array, mean, ones
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM, multiply, subtract, add, Activation, Lambda, Flatten, maximum
from keras.layers import Dense, concatenate, MaxPooling1D, LocallyConnected1D, Reshape, Dropout
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import constraints

import tensorflow as tf

from utils import generate_student_name
import random

from matplotlib import pyplot as plt

n_traits = 2


# In[2]:


generate_student_name()


# In[3]:


from keras import backend as K
from keras.constraints import Constraint
from keras.engine.topology import Layer
from keras import initializers

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, min_w=0, max_w=4):
        self.min_w = min_w
        self.max_w = max_w

    def __call__(self, p):
        return K.clip(p, self.min_w, self.max_w)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'min_w': self.min_w,
                'max_w': self.max_w }


class ProductLayer(Layer):

    def __init__(self, output_dim, kernel_constraint=WeightClip(min_w=-4.0, max_w=4.0), minv=-4,maxv=4, **kwargs):
        
        self.output_dim = output_dim
        super(ProductLayer, self).__init__(**kwargs)
        self.kernel_constraint= constraints.get(kernel_constraint)
        self.min_v = minv
        self.max_v = maxv

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, self.output_dim),
                                      initializer=initializers.RandomUniform(minval=self.min_v,maxval=self.max_v),
#                                       initializer=initializers.Constant(value=2.0),
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        
        super(ProductLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        p = x * self.kernel
        print("shape p", p.shape)
        return p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class DifferenceLayer(Layer):

    def __init__(self, output_dim, kernel_constraint=WeightClip(min_w=-4.0, max_w=4.0), minv=-4,maxv=4, invert=False, **kwargs):
        
        self.output_dim = output_dim
        super(DifferenceLayer, self).__init__(**kwargs)
        self.kernel_constraint= constraints.get(kernel_constraint)
        self.min_v = minv
        self.max_v = maxv
        self.invert = invert

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        initialiser = initializers.RandomUniform(minval=self.min_v,maxval=self.max_v)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, self.output_dim),
                                      initializer=initialiser,
#                                       initializer=initializers.Constant(value=2.0),
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        super(DifferenceLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.invert:
            x = tf.Print(x, [x], message="x is:", first_n=-1, summarize=1024)
            k = tf.Print(self.kernel, [self.kernel], message="- kernel is:", first_n=-1, summarize=1024)
            p = x - k
        else:
            k = tf.Print(self.kernel, [self.kernel], message="kernel is:", first_n=-1, summarize=1024)
            x = tf.Print(x, [x], message="- x is:", first_n=-1, summarize=1024)
            p = k - x
#         p = K.print_tensor(p, message="p is:")
        p =  tf.Print(p, [p], message="p is:", first_n=-1, summarize=1024)
        print("shape p", p.shape)
        return p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# In[4]:


for z in [-20, -10, -4,-3,-2,-1,0,1,2,3,4]:
    print(z, 1/(1+exp(-z)) )

# q_p_avg = 0.45
q_p_easiest = 0.9999
q_p_hardest = 0.0001
mid = (q_p_easiest + q_p_hardest)/2.0

# pr_k_avg = q_p_avg**(1/n_traits)
# print("pr k avg:", pr_k_avg)

pr_k_easiest = q_p_easiest**(1/n_traits)
pr_k_hardest = q_p_hardest**(1/n_traits)
pr_k_mid = mid**(1/n_traits)

inv_sigmoid = lambda pr : ( -log((1/pr) -1) )
easy_comp_del = inv_sigmoid(pr_k_easiest)
hard_comp_del = inv_sigmoid(pr_k_hardest)

offset = (easy_comp_del - hard_comp_del)/2
beta_min = 0
beta_max = round(offset,1)
theta_min = round(easy_comp_del - offset,1)
theta_max = round(easy_comp_del,1)

# beta_min = 0
# beta_max = 10
# theta_min = 5
# theta_max = 15


print(beta_min, beta_max)
print(theta_min, theta_max)

worst_comp_pr = 1/(1+exp(-(theta_min - beta_max)))
best_comp_pr = 1/(1+exp(-(theta_max - beta_min)))

print("worst cmp chance=", worst_comp_pr)
print("best cmp chance=", best_comp_pr)

print("worst Pr=", worst_comp_pr**n_traits)
print("best Pr=", best_comp_pr**n_traits)


nom = array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
])

sel = nom[[0,2]]
print(sel)
print(sel.shape)


# In[5]:


class BigTable(Layer):

    def __init__(self, _dim, min_w=0, max_w=10, **kwargs):
        self.dim = _dim
        self.limits = (min_w, max_w)
        kc =WeightClip(min_w, max_w)
        self.kernel_constraint= constraints.get(kc)
        super(BigTable, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        min_w, max_w = self.limits
        av_w = (min_w + max_w)/2.0
        initialiser = initializers.RandomUniform(min_w, max_w)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.dim),
                                      initializer=initialiser,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        print("kk", self.kernel.shape)
        super(BigTable, self).build(input_shape)  # Be sure to call this at the end

    def call(self, selector):
        print("selector shape", selector.shape)
        selector = K.flatten(selector)
        print("flat selector shape", selector.shape)
        print("call kk", self.kernel.shape)
        # selector = tf.Print(selector, [selector], message="selector is:", first_n=-1, summarize=1024)
        rows = K.gather(self.kernel, selector)
#         rows = tf.Print(rows, [rows], message="row is:", first_n=-1, summarize=1024)
        print("'rows' shape,",rows.shape)
        return rows

    def compute_output_shape(self, input_shape):
        return ((None, self.dim[1]))


# In[6]:


class Question():
    def __init__(self, qix, min_diff, max_diff, nt=None, nnw=None, optimiser=None):
        #self.MAX_BETA = 15
        self.id = qix
#         no_dummies = randint(1,(nt-1))
        no_live = nt
#         print("no_dummies=",no_dummies)
        not_present= -10
#         min_diff = 0
#         max_diff = 10
        self.betas = [ not_present for _ in range(nt) ]
        choices = random.sample(range(nt), no_live)
        for c in choices:
            self.betas[c] = round(random.randint(10*min_diff,10*max_diff)/10,1)
#             self.betas = [ round(random.randint(10*min_diff,10*max_diff)/10,1) for _ in range(nt)]


# In[7]:


class Student():
    def __init__(self, psix, min_abil, max_abil, nt=None, nnw=None, optimiser=None):
        #self.MAX_BETA = 15
        self.id = psix
        self.name = generate_student_name()
#         min_abil = 0
#         max_abil = 10
        self.thetas = [ round(random.randint(10*min_abil, 10*max_abil)/10,1) for _ in range(nt) ]
#         self.mastery = [0 for _ in range(nq)]
#         self.o_practice = [0 for _ in range(nq)]
#         self.h_practice = [0 for _ in range(nt)]
        #print("Made q with betas:", self.betas)


# In[8]:


def attempt_q(student: Student, q: Question):
    p = calculate_pass_probability(student.thetas, q.betas)
    this_att = uniform(0,1)
    if (this_att <= p):
        passed=1
        print("passed")
#         student.mastery[q.id] = 1
    else:
        passed=0

    return passed


# In[9]:


def calculate_pass_probability(thetas, betas):
    # additive factors model is:
    # p_pass = 1 / 1 + exp(-z)
    # where z = a + sum[1:n]( -b + gT )
    
    p_pass = 1.0
#     print("th,b",thetas,betas)
    for th,b in zip(thetas,betas):
        z = (th-b)
        p_pass_step = 1.0 / (1.0 + exp(-z))
#         print(th,"vs",b,"->",z,": ", p_pass_step)
        p_pass *= p_pass_step # simple conjunctive model of success!
    try:
        pass
        print("p_pass={}".format(p_pass))
    except OverflowError:
        p_pass = 0.0
#     print("real p_pass = {}".format(p_pass))
    return p_pass
    


# In[10]:


qopt = Adam()

def create_qs(n_qs, nt=n_traits, nnw=n_traits, optimiser=qopt):
    random.seed(666)
    master_qs = [Question(qix, beta_min,beta_max, nt=nt, nnw=nnw, optimiser=optimiser) for qix in range(n_qs)]
    for q in master_qs:
        nocomps = len(q.betas)
        mag = sqrt(sum([ pow(b, 2) for b in q.betas if b!=-10 ]))
        print("Q:{}, difficulty={:.2f} across {} components".format(q.id, mag, nocomps))
    
    for q in master_qs:
        print("qid",q.id,q.betas)
    
    qn_table = BigTable((n_qs, nnw),min_w=beta_min, max_w=beta_max)
    
    return master_qs, qn_table


# # Training
# This is where sh!t gets real.  We take our tr_len (1000?) students, and iterate over them 100 times to create 100,000 *complete examples* of a student attacking the curriculum.  The questions themselves are attacked in random order: the student has no intelligent guidance throught the material. (Obvious we may wish to provide that guidance at some point in the future.)
# 
# Remember, there are only 12 exercises in the curriculum, so if the student is taking 60 or 70 goes to answer them all, that's pretty poor.  But some of these kids are dumb as lumber, so cut them some slack!  They will all get there in the end since by the CMU AFM practice will, eventually, make perfect!

# In[11]:


psi_opt = Adam()
def create_students(n_students, nt=n_traits, nnw=n_traits, optimiser=None):
    random.seed(666)
    psi_list = [ Student(psix, theta_min,theta_max, nt=nt, nnw=nnw, optimiser=optimiser) for psix in range(n_students)]
    for psi in psi_list:
        print(psi.name, psi.thetas)
        
    psi_table = BigTable((n_students, nnw), min_w=theta_min, max_w=theta_max)
    psi_attn_table = BigTable((n_students, 1), min_w=0, max_w=1)
    print("psi_table wgts", psi_table.get_weights())
    
    return psi_list, psi_table
    


# In[12]:


extend_pop=False
extend_by = 90
if extend_pop:
    for _ in range(extend_by):
        nu_psi = Student(nt=n_traits, nq=len(master_qs), optimiser=psi_opt)
        psi_list.append(nu_psi)


# In[13]:


import gc
def generate_attempts(master_qs, psi_list):
    attempts =[]
    attempts_by_q = {}
    attempts_by_psi = {}

    user_budget = math.inf
    user_patience = 10
    pass_to_remove = True
    for run in range(1):
        print("----{}\n".format(run))
        for psi in psi_list:
            spend=0
            #psi.mastery = [0 for _ in range(nq)]
            qs = [ix for ix in range(len(master_qs))]
#             print("* * * **** USER {}".format(psi.name))
#             print("* * * * ** THETAS {}".format(psi.thetas))

            while(True):
                q_ct = 0
                qix = random.choice(qs)
                q = master_qs[qix]
                passed=0

                if psi.name not in attempts_by_psi:
                    attempts_by_psi[psi.name]=[]

                if q not in attempts_by_q:
                    attempts_by_q[q]=[]

                while (not passed) and q_ct<user_patience:
                    q_ct+=1
                    passed = attempt_q(psi, q)
                
                tup = (psi.id, q.id, passed, (q_ct if passed else 0))
                attempts.append(tup)
                attempts_by_psi[psi.name].append(tup)
                attempts_by_q[q].append(tup)
                                        
                if passed:
                    qs.remove(qix)

                spend += q_ct

                if qs == [] or spend>=user_budget:
                    break
    gc.collect()
    return attempts, attempts_by_q, attempts_by_psi



# def generate_model(qn_table, psi_table, optimiser, mode="q_train"):
#     psi_sel = Input(shape=(1,), name="psi_select", dtype="int32")
#     qn_sel = Input(shape=(1,), name="q_select", dtype="int32")
#     print(qn_table, psi_table, psi_sel, qn_sel)
#     print("psi_sel shape", psi_sel.shape)

#     if mode=="q_train":
#         psi_table.trainable=False
#         qn_table.trainable=True
#     else:
#         psi_table.trainable=True
#         qn_table.trainable=False
    
#     qn_row = qn_table(qn_sel)
#     psi_row = psi_table(psi_sel)
# # #     qn_row = Lambda(lambda q: q[:,qn_sel,:])(qn_table)
# # #     psi_row = Lambda(lambda s: s[:,psi_sel,:])(psi_table)
# #     print("shape qn_row",qn_row.shape)
# #     print("shape psi row", psi_row.shape)
# #     qn_row = Lambda(lambda q: tf.Print(q, [q], message="qn row is:", first_n=-1, summarize=1024))(qn_row)
# #     psi_row = Lambda(lambda q: tf.Print(q, [q], message="psi row is:", first_n=-1, summarize=1024))(psi_row)
# #     qn_row = Reshape((1,))(qn_row)
# # #     psi_row = Reshape((1,))(psi_row)
    
# # #     qn_row = Flatten()(qn_row)
# # #     psi_row = Flatten()(psi_row)
# #     print("psi row shape", psi_row.shape)
# #     print("qn_eow shape", qn_row.shape)
# #     dif = subtract([psi_sel, qn_sel])
#     dif = subtract([psi_row, qn_row])
# #     dif = concatenate([qn_row, psi_row])
# #     dif = Lambda(lambda a: a[0] - a[1])([psi_row, qn_row])
#     print("dif",dif.shape)
#     Prs = Lambda(lambda z: 1.0 / (1.0 + K.exp(-z)), name="sPr_sigmoid")(dif)
#     print("Prs",Prs.shape)
#     Pr = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="sPr_prod")(Prs)
#     print("Pr",Pr.shape)
#     model = Model(inputs=[qn_sel, psi_sel], outputs=Pr)
#     model.compile(optimizer=optimiser, loss="mean_absolute_error", metrics=["mse","accuracy"])
#     return model
    
def generate_qs_model(qn_table, psi_table, optimiser):
    psi_sel = Input(shape=(1,), name="psi_select", dtype="int32")
    qn_sel = Input(shape=(1,), name="q_select", dtype="int32")
    did_pass = Input(shape=(1,), name="did_pass")
    did_n = Input(shape=(1,), name="did_n")
    print(qn_table, psi_table, psi_sel, qn_sel)
    print("psi_sel shape", psi_sel.shape)

    psi_table.trainable=True
    qn_table.trainable=True
    
    qn_row = qn_table(qn_sel)
    psi_row = psi_table(psi_sel)
    dif = subtract([psi_row, qn_row])
    print("dif",dif.shape)
    Prs = Lambda(lambda z: K.exp(z) / (1.0 + K.exp(z)), name="sPr_sigmoid")(dif)
    
    print("Prs",Prs.shape)
    #calculate the hazard rate here
    hz = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="sPr_prod")(Prs)
#     print("hz",hz.shape)
    
    max_att = 10
    #calculate the prob of passing in <= max_att
    #prob of passing at point t:
    #Pr(X=t) = (1-p)^(t-1) * p # i.e. fail^(t-1), then pass 
    
#     Pr = (1-p)^(t-1) * p
    
    
#     hidden = Dense(5, activation="relu")(Prs)
#     hidden = Dense(5, activation="relu")(hidden)
#     hidden = Dense(5, activation="relu")(hidden)
#     n = Dense(1, activation="relu")(hidden)
    n = Lambda(lambda p: (1/p), name="n_calc")(hz)
    print("n",n.shape)
#     did_pass_flat = Lambda(lambda i: K.flatten(i))(did_pass)
    n = multiply([n, did_pass], name="n_cancellor")

    print("did_pass", did_pass.shape)
    print("n'",n.shape)

#     pazz= hz
#     hz = Lambda(lambda hz: tf.Print(hz, [hz], message="hz is:", first_n=-1, summarize=1024))(hz)
    pazz = Lambda( lambda p : (1.0-K.pow((1-p),max_att)), name="pazz_calc" ) (hz)
    # pazz = Lambda(lambda t: tf.Print(t, [t], message="pazz is:", first_n=-1, summarize=1024))(pazz)
#     hidden2 = Dense(5, activation="relu")(Prs)
#     hidden2 = Dense(5, activation="relu")(hidden2)
#     hidden2 = Dense(5, activation="relu")(hidden2)
#     pazz = Dense(1, activation="sigmoid")(hidden2)
    
    print("pazz",pazz.shape)
    
    model = Model(inputs=[qn_sel, psi_sel, did_pass], outputs=[pazz,n])
    model.compile(optimizer=optimiser, loss=["binary_crossentropy","mae"], metrics=["accuracy"], loss_weights=[1.0,1.0])
    return model


# In[14]:


def init_weights(master_qs, psi_list, attempts_by_psi, attempts_by_q, q_table, s_table, max_b, min_th):
    psi_wgts = s_table.get_weights()[0]
    for s in psi_list:
#         attz = [tup[2] for tup in attempts_by_psi[s.name]]
#         prop = mean(attz)
#         p = prop**(1/n_traits)
#         cw_prop = log(p / (1-p))
        psi_wgts[s.id,:] = numpy.random.uniform(min_th,min_th+0.01, size=psi_wgts.shape[1])
        print(psi_wgts[s.id,:])
    s_table.set_weights([ psi_wgts ])

    qn_wgts = q_table.get_weights()[0]
    for q in master_qs:
#         attz = [tup[2] for tup in attempts_by_q[q]]
#         prop = mean(attz)
#         p = prop**(1/n_traits)
#         cw_prop = log((1-p) / p)
        qn_wgts[q.id,:]= numpy.random.uniform(max_b-0.01,max_b, size=qn_wgts.shape[1])
        print(qn_wgts[q.id,:])
    q_table.set_weights([ qn_wgts ])
    


# In[15]:


import os
# import IPython

def calibrate(master_qs, psi_list, qs_model, q_model, s_model, attempts, attempts_by_psi, attempts_by_q, psi_wgts, qn_wgts, n_iter=20, record_param_fit=False):
    es = EarlyStopping(monitor="val_loss", mode="auto", patience=0)
    random.seed(666)
    min_mse = 1000
    min_avg_fit_rmse = math.inf
    min_loss= math.inf
    q_outer_mses = []
    q_outer_accs = []
    s_outer_mses = []
    s_outer_accs = []
    th_mses = []
    b_mses = []
    th_accs= []
    b_accs =[]
    h= []
    avg_fit_rmses = []
    th_fit_rmses = []
    b_fit_rmses = []
    init_patience = 100
    patience = init_patience
    last_avg_fit_rmse = math.inf
    
    op_loop = 100

#     attempts = attempts
#     qices = array([int(tup[1]) for tup in attempts]).flatten() #reshape(-1,1)
#     psices = array([int(tup[0]) for tup in attempts]).flatten() #reshape(-1,1)
#     pfs = array([tup[2] for tup in attempts]).flatten() #reshape(-1,1)
#     len_all = qices.shape[0]
    
#     att_ct = defaultdict(float)
#     passmap = {}
#     for tup in attempts:
#         psi = tup[0]
#         q = tup[1]
#         pf = tup[2]
#         att_ct[(psi,q)] = 1.0+att_ct[(psi,q)]
#         passmap[(psi,q)] = pf
            
#     for k in att_ct.keys():
#         print(k)
#         if k in passmap:
#             print(k,"in passmap")
#             att_ct[k] = 1.0/(att_ct[k])
#         else:
#             att_ct[k] = 0.0
        
#     attempts_by_q = defaultdict(list)
#     attempts_by_psi = defaultdict(list)
#     attempts = []
    
#     print("passmap")
#     print(passmap)
#     for q in master_qs:
#         for psi in psi_list:
#             ct = att_ct[(psi.id,q.id)] 
# #             prob = att_ct[(psi.id,q.id)]
#             if passmap[(psi.id, q.id)]==1:
#                 #prob = (1.1/(ct*1.1))**(1/ct)
#                 prob = 1 #(1/ct) #**(1/ct)
# #                 attz = [ uniform(0,1)<(1/ct) for _ in range(100) ]
# #                 prob = mean(attz)                    
#             else:
#                 prob = 0
        
#             if (prob < 0.05):
#                 prob = 0.05
#             elif prob > 0.8:
#                 prob = 0.8
#             tvp = (psi.id, q.id, prob)
#             print("tvp=",tvp)
#             attempts.append(tvp)
#             attempts_by_q[q.id].append(tvp)
#             attempts_by_psi[psi.id].append(tvp)

    print(attempts[0:100])
#     input("churl")

    qices = array([int(tup[1]) for tup in attempts]).flatten() #reshape(-1,1)
    psices = array([int(tup[0]) for tup in attempts]).flatten() #reshape(-1,1)
    pfs = array([tup[2] for tup in attempts]).flatten() #reshape(-1,1)
    ns = array([tup[3] for tup in attempts]).flatten() #reshape(-1,1)
    len_all = qices.shape[0]
        
#     zz = array([0 for tup in attempts]).flatten()
        
    qz = qices
    sz = psices
    pfz = pfs
    nz = ns

    print(qz)
    print(sz)
    print(pfz)
    print(nz)

    loss = False
    mse = False
    acc = False
    early_stop = True
    min_stop = 0
#     n_iter = 1
    for i in range(n_iter):
#         shuffle(attempts)
#         attemptz = attempts[0:100]
#         qices = array([int(tup[1]) for tup in attemptz]).flatten() #reshape(-1,1)
#         psices = array([int(tup[0]) for tup in attemptz]).flatten() #reshape(-1,1)
#         pfs = array([tup[2] for tup in attemptz]).flatten() #reshape(-1,1)
#         len_all = qices.shape[0]      

#         qz = qices
#         sz = psices
#         pfz = pfs
        
#         psi = random.choice(psi_list)
#         q = random.choice(qs)
#         print(psi.id, psi.name, q.id)
#     while True:
#         numpy.random.shuffle(attempts)
        base_ix = 0
        done = False
#         for j in range(chunkz+1):
        j = 0
        sub_h = []
        
#         attempts = attempts_by_q[q.id]
#         qz = array([int(tup[1]) for tup in attempts]).flatten()#.reshape(-1,1)
#         sz = array([int(tup[0]) for tup in attempts]).flatten()#.reshape(-1,1)
#         pfz = array([int(tup[2]) for tup in attempts]).flatten()#.reshape(-1,1)
#         s_table.trainable = False
#         q_table.trainable = True
#         q_model.train_on_batch(x=[qz, sz], y=pfz)#, epochs=10, shuffle=True, batch_size=1, callbacks=[es])
#         q_model.fit(x=[qz, sz], y=pfz, epochs=1, shuffle=True, batch_size=1, callbacks=[es], verbose=1)

#         attempts = attempts_by_psi[psi.id]
#         qz = array([int(tup[1]) for tup in attempts]).flatten()#.reshape(-1,1)
#         sz = array([int(tup[0]) for tup in attempts]).flatten()#.reshape(-1,1)
#         pfz = array([int(tup[2]) for tup in attempts]).flatten()#.reshape(-1,1)
#         s_table.trainable = True
#         q_table.trainable = False
        qs_model.train_on_batch(x=[qz, sz, pfz], y=[pfz,nz])#, epochs=10, shuffle=True, batch_size=1, callbacks=[es])
#         qs_model.fit(x=[qz, sz, pfz], y=[pfz,nz], epochs=1, shuffle=True, batch_size=100) #, callbacks=[es])

#         qs_model.fit(x=[qz, sz], y=pfz, epochs=100, shuffle=True, validation_split=0.1, batch_size=len(pfz), callbacks=[es], verbose=1)
        k=15
        mr_hat = qs_model.predict(x=[qz[0:k], sz[0:k], pfz[0:k]])
#         for el in mr_hat:
#             print(el)
        if i%op_loop==0:
            for s,q,pf,n,hat1,hat2 in zip(sz[0:k], qz[0:k], pfz[0:k], nz[0:k], mr_hat[0][0:k],mr_hat[1][0:k]):
                print(s,q,pf,n,":",float(hat1),float(hat2))

        if i % 1 == 0:
            loss_tup = qs_model.evaluate(x=[qices, psices, pfs], y=[pfs,nz], verbose=0) #, epochs=1, shuffle=True, batch_size=1, verbose=0) #, callbacks=[es])
            comb_loss, clf_loss, n_loss, acc, _ = loss_tup
#             loss2, mse2, acc2 = s_model.evaluate(x=[qices, psices], y=pfs, verbose=0) #, epochs=1, shuffle=True, batch_size=1, verbose=0) #, callbacks=[es])
#             loss=(loss+loss2)/2
#             mse=(mse+mse2)/2
#             acc=(acc+acc2)/2
            sub_h.append((comb_loss, clf_loss, n_loss, acc))

            psi_wgts = s_table.get_weights()[0]
            th_rmses = []
            for s in psi_list:
                s_thetas = numpy.sort(s.thetas)
                s_wgts = numpy.sort(psi_wgts[s.id])
                err = numpy.abs(s_thetas - s_wgts)
#                 err = cosine(s_thetas, s_wgts)
                th_rmses.append(err)

            qn_wgts = q_table.get_weights()[0]
            b_rmses = []
            for q in master_qs:
                q_betas = numpy.sort(q.betas)
                q_wgts = numpy.sort(qn_wgts[q.id])
                err = numpy.abs(q_betas - q_wgts)
#                 err = cosine(q_betas, q_wgts)
                b_rmses.append(err)
            
            th_rmse = numpy.mean(th_rmses)
            th_fit_rmses.append(th_rmse)
            if i % op_loop == 0:
                print("i =",i)
                print("th RMSE=", th_rmse)
            b_rmse = numpy.mean(b_rmses)
            b_fit_rmses.append(b_rmse)
            if i % op_loop == 0:
                print("b RMSE=", b_rmse)
                print("losses:",loss_tup)
            sub_h = numpy.array(sub_h)
            sub_tup = (sub_h[-1,0], sub_h[-1,1], sub_h[-1,2], sub_h[-1,3])
            h.append(sub_tup)
#             av_rmse = (th_rmse/len(psi_list) +b_rmse/len(master_qs))
#             av_rmse =(th_rmse+b_rmse)/2.0
            av_rmse = b_rmse
            if i % op_loop == 0:
                print("av RMSE=",av_rmse)
#             if av_rmse < last_avg_fit_rmse:
            if av_rmse < min_avg_fit_rmse:
                patience = init_patience
                if i % op_loop == 0:
                    print("patience reset to", init_patience)
                #if av_rmse < min_avg_fit_rmse:
                    min_avg_fit_rmse = av_rmse
                    min_loss = loss
                    qs_model.save_weights("qs_best_weights.hdf5")
            else:
                if early_stop and i>min_stop:
#                 if loss < min_loss:
#                     print(loss,"<",min_loss)
                    if patience >0:
                        patience -= 1
                        print("patience now", patience)
                    else:
                        print("Earlying stoppin' @",i)
                        break
            last_avg_fit_rmse = av_rmse
                    
    #     del h
    #     loss, mse, acc = qs_model.evaluate(x=[qices, psices], y=pfs)

    #     print(loss, mse, acc)
    return h, th_fit_rmses, b_fit_rmses


# In[16]:


nn_dimensions = [n_traits]
serieses = []
min_errs = []
n_qs = 25
n_students = 1000
opt = Adam(lr=0.5) #try 0.5 for 2 dim
for ix,nnw in enumerate(nn_dimensions):
    qs, q_table = create_qs(n_qs, n_traits, nnw, optimiser=opt)
    ss, s_table = create_students(n_students, n_traits, nnw, optimiser=opt)
    attempts, attempts_by_q, attempts_by_psi = generate_attempts(qs,ss)
    
    pf = [tup[2] for tup in attempts]
    N = len(pf)
    pN = sum(pf)
    pr = pN/N
#     input("pass rate is {} of {} = {}".format(pN,N,pr))
    
#     print(attempts)
#     q_model = generate_model(q_table, s_table, opt, mode="q_train")
#     s_model = generate_model(q_table, s_table, opt, mode="s_train")
    q_model = None
    s_model = None
    qs_model = generate_qs_model(q_table, s_table, opt)
#     qs_model = None
#     psi_model = generate_psi_model(s_table, qopt)
    init_weights(qs, ss, attempts_by_psi, attempts_by_q, q_table, s_table, beta_max, theta_min)
#     input("chunt")
    psi_wgts = s_table.get_weights()[0]
    qn_wgts = q_table.get_weights()[0]
    h, th_fit_rmses, b_fit_rmses = calibrate(qs,ss, qs_model, q_model, s_model, attempts, attempts_by_psi, attempts_by_q, psi_wgts, qn_wgts, n_iter=1000000, record_param_fit=True)
    qs_model.load_weights("qs_best_weights.hdf5")


# In[ ]:


print("elements in h:", len(h))
for tup in h:
    print(tup)

comb_loss, clf_loss, n_loss, acc = zip(*h)


fig = plt.gcf()
#     plt.xlabel("#iterations")
#     plt.ylabel("fit error (RMSE)")
#     plt.suptitle("Neural-MLTM Parameter Fitting")
#     plt.title("(skills=5, items=10, students=100)")
fig, axes = plt.subplots(nrows=1, ncols=4)
axes[0].plot(comb_loss, label="comb_loss")
axes[1].plot(clf_loss, label="clf_loss")
axes[2].plot(n_loss, label="n_loss")
axes[3].plot(acc, label="acc")

fig.set_size_inches(20, 5)
for i in [0,1,2,3]:
    axes[i].legend()
plt.show()

av_fit_rmses = []
for b,th in zip(b_fit_rmses, th_fit_rmses):
    av = (b+th)/2.0
    av_fit_rmses.append(av)
    
plt.plot(b_fit_rmses, label="beta fit")
plt.plot(th_fit_rmses, label="theta fit")
plt.plot(av_fit_rmses, label="av")
plt.legend()
plt.show()
# fig = plt.gcf()
# fig.set_size_inches(8, 5)
# plt.xlabel("#iterations")
# plt.ylabel("fit error (RMSE)")
# plt.suptitle("Neural-MLTM Parameter Fitting")
# plt.title("(skills=5, items=10, students=100)")
# plt.legend()
# plt.show()


# In[ ]:


from scipy.spatial.distance import cosine

real_wgts = array([ q.betas for q in qs ])
pred_wgts = q_table.get_weights()[0]
# pred_wgts = numpy.round(pred_wgts,1)

out_cols = [None] * len(real_wgts.T)
curr_sel = None
curr_ix = None
n_iters = 10
chosen = None

indices = range(len(real_wgts.T))

min_total_err = math.inf
for i in range(10): #len(indices)**2):
    print("i is ",i)
    real_used = set()
    pred_used = set()
    while len(pred_used) < len(indices):
        curr_mse = math.inf
        for rix in numpy.random.permutation(indices):
            if rix in real_used:
                continue
            real_col = real_wgts.T[rix]
            for cix in numpy.random.permutation(indices):
                if cix in pred_used:
                    continue
                pred_col = pred_wgts.T[cix]
                mse = numpy.mean(numpy.abs( pred_col - real_col))
#                 mse = cosine(pred_col, real_col)
                #print("mae is ",mse)
                if mse < curr_mse:
#                     print("best match", cix, rix)
#                     print(real_col)
#                     print(pred_col)
                    curr_sel = pred_col
                    curr_mse = mse
                    curr_ix = cix
                    curr_real_ix = rix
#         print("---")
        real_used.add(curr_real_ix)
        pred_used.add(curr_ix)
        out_cols[curr_real_ix] = curr_sel
    out_col_arr = array(out_cols).T
    total_err = numpy.mean(numpy.abs( out_col_arr - real_wgts ))
#     total_err = cosine(out_col_arr.flatten(), real_wgts.flatten())
    mean_ll = numpy.mean( out_col_arr - real_wgts )
    if total_err < min_total_err:
        min_total_err = total_err
        best_ll = mean_ll
        chosen = out_col_arr
        print("new total min mae:", min_total_err)
        print("new best ll", best_ll)
        
print("real", real_wgts)
# print(pred_wgts)
print("out", chosen)
print("elementwise mae:", min_total_err)
print("mean lead/lag", mean_ll)


# In[ ]:


#print(itemz.shape)


# In[ ]:


fig = plt.gcf()
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
pca2 = PCA(n_components=2)
# pca = TSNE(n_components=2)
# pca2 = TSNE(n_components=2)

itemz = array([ q.betas for q in qs ])

itemz_2 = itemz
# itemz_2 = pca.fit_transform(itemz_2)
# itemz_2 = MinMaxScaler().fit_transform(itemz_2)

itemz_pred = chosen
# itemz_pred = pca.transform(itemz_pred)
# itemz_pred = MinMaxScaler().fit_transform(itemz_pred)
# print(itemz_2)

# fig,axs = plt.subplots(1,2)
fig = plt.gcf()
fig.set_size_inches(15, 10)

for x,xh,y,yh in zip(itemz_2[:,0],itemz_pred[:,0],itemz_2[:,1],itemz_pred[:,1]):
    fig.gca().plot([x,xh],[y,yh],color="#aaaaaa")

fig.gca().scatter(itemz_2[:,0], itemz_2[:,1], alpha=0.7)
fig.gca().scatter(itemz_pred[:,0], itemz_pred[:,1], alpha=0.5)

for i, txt in enumerate(itemz_2):
    fig.gca().annotate(i, (itemz_2[i,0], itemz_2[i,1]))

for i, txt in enumerate(itemz_pred):
    fig.gca().annotate(i, (itemz_pred[i,0], itemz_pred[i,1]))
    
fig.show()

