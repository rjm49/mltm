from importlib import reload

from collections import defaultdict

import numpy

import random
from matplotlib import pyplot as plt
from random import shuffle, choice, randint

import math

from NN_utils import BigTable, WeightClip

import pickle
import zlib

def logistic(x, b,off):
    z = b*(x-off)
    return numpy.exp(z)/(1+numpy.exp(z))

def pr_to_spread(p, comps=1, as_A_and_D=True):
    per_comp_p = p**(1/(comps))
#     print("p         ", p)
#     print("per comp p", per_comp_p)
#     spread = -numpy.log((1.0/per_comp_p)-1.0)
    inv_sigmoid = lambda pr : ( -numpy.log((1/pr) -1) )
    spread = inv_sigmoid(per_comp_p)
#     print("spread    ", spread)
    if as_A_and_D:
        a = spread/2.0
        d = -spread/2.0
        return a,d
    else:
        return spread

print("started")


# In[2]:


from numpy.random import uniform, random_integers
from scipy.stats import truncnorm

def gen_bayes_students_questions(n_students, n_questions, a0, a1, n_factors, min_active_traits, max_active_traits):
    students = numpy.zeros((n_students, n_factors))
    for six in range(n_students):
#         true_comps = numpy.random.normal(a0, a1, size=n_factors)#+0.2#+2#+1.2
#         true_comps = numpy.random.normal(0, 5/3, size=n_factors)#+0.2#+2#+1.2
        true_comps = numpy.random.uniform(0,1, size=n_factors)#+0.2#+2#+1.2
        for cix,c in zip(range(n_factors), true_comps):
            students[six,cix] = c

    questions = numpy.zeros((n_questions, n_factors))
    for qix in range(n_questions):
#         n_comps = randint(min_active_traits, max_active_traits)
        n_comps = n_factors
        comp_ixs = numpy.random.choice(range(n_factors), size=n_comps, replace=False)
        true_comps = numpy.random.uniform(a0[0],a0[1], size=n_comps)

        for cix,c in zip(comp_ixs,true_comps):
            questions[qix,cix] = c#+boost
    
    print("genqs",students.shape, questions.shape)
    return students, questions

def gen_bayes_run(n_traits, a0, a1, min_active_traits, max_active_traits, test_w=None, n_students=8, n_questions=8):
    students, questions = gen_bayes_students_questions(n_students, n_questions, a0, a1, n_factors, min_active_traits, max_active_traits)
    print(students.shape, questions.shape)
    obs = numpy.zeros((len(students), len(questions)))
    probs = numpy.zeros((len(students), len(questions)))
    vz = []
    mz = []
    scz =[]
    for vi in range(len(students)):
        for mi in range(len(questions)):
            prs = (1-questions[mi]) + (questions[mi]*students[vi])
            pr = numpy.prod(prs)
            obs[vi,mi] = (random.random() < pr)
            probs[vi,mi] = pr
    return obs, probs, students, questions

# n_students, n_questions, n_factors, min_active, max_active = 100,100,10,10,10


# In[3]:


from numpy.random import uniform, random_integers
from scipy.stats import truncnorm

def gen_rasch_students_questions(n_students, n_questions, a0, a1, n_factors, min_active_traits, max_active_traits, test_w):
    students = numpy.zeros((n_students, n_factors))
    for six in range(n_students):
        true_comps = numpy.random.normal(0, a1, size=n_factors)#+0.2#+2#+1.2
#         true_comps = numpy.random.normal(0, 5/3, size=n_factors)#+0.2#+2#+1.2
#         true_comps = numpy.random.uniform(0,1, size=n_factors)#+0.2#+2#+1.2
        for cix,c in zip(range(n_factors), true_comps):
            students[six,cix] = c

    av_c = (min_active_traits + max_active_traits)/2
    d50 = pr_to_spread(.5, av_c, as_A_and_D=False)
    
    questions = numpy.zeros((n_questions, n_factors)) - 50
    
    minb=-(test_w/2) -a0 - d50
    maxb=(test_w/2) -a0 - d50
    questions = questions
    minb, maxb = sorted([minb, maxb])
    minb = float(minb)
    maxb = float(maxb)
    
    for qix in range(n_questions):
        n_comps = randint(min_active_traits, max_active_traits)
#         print("n_comps", n_comps)
        comp_ixs = numpy.random.choice(range(n_factors), size=n_comps, replace=False)  
#         print("range=", minb,maxb)
        true_comps = numpy.random.uniform(minb, maxb, size=n_comps)
        for cix,c in zip(comp_ixs,true_comps):
            questions[qix,cix] = c
            
    return students, questions

def gen_rasch_run(n_traits, a0, a1, min_active_traits, max_active_traits, test_w=None, n_students=8, n_questions=8):
    students, questions = gen_rasch_students_questions(n_students, n_questions, a0, a1, n_factors, min_active_traits, max_active_traits, test_w=test_w)
    obs = numpy.zeros((len(students), len(questions)))
    probs = numpy.zeros((len(students), len(questions)))
    vz = []
    mz = []
    scz =[]
    for vi in range(len(students)):
        for mi in range(len(questions)):
#             zmask = (questions[mi] < 0.001).astype(int)
            diffs = students[vi]-questions[mi]
            prs = logistic(diffs,1,0)
#             prs = numpy.maximum(zmask,prs)

            pr = numpy.prod(prs)
            obs[vi,mi] = (random.random() < pr)
            probs[vi,mi] = pr
    return obs, probs, students, questions


from keras.regularizers import l1
from keras.layers import Dropout, multiply, subtract, GaussianNoise, GaussianDropout, Input, Lambda
from keras import backend as K, Model
def generate_qs_model(qn_table, psi_table, optimiser, _mode="MXFN", loss="MSE"):
    psi_sel = Input(shape=(1,), name="psi_select", dtype="int32")
    qn_sel = Input(shape=(1,), name="q_select", dtype="int32")
    print(qn_table, psi_table, psi_sel, qn_sel)
    print("psi_sel shape", psi_sel.shape)

    psi_table.trainable=True
    qn_table.trainable=True
    
    qn_row = qn_table(qn_sel)
    psi_row = psi_table(psi_sel)

    print("Mode is", _mode)
    if _mode=="COND":
        Prs = Lambda(lambda qs: (1 - qs[0])+(qs[0]*qs[1]), name="Prs")([qn_row, psi_row])
        score = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="score")(Prs)
    if _mode=="BINQ":
        Prs = Lambda(lambda qs: (1 - qs[0])+(qs[0]*qs[1]), name="Prs")([qn_row, psi_row])
        score = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="score")(Prs)
    elif _mode=="MLTM":
#         klip = Lambda(lambda qk: 10*(K.clip(qk,-5, -4.9)+5))
#         q_masque = klip(qn_row)
        difs = subtract([psi_row, qn_row])
        Prs = Lambda(lambda z: (1.0 / (1.0 + K.exp(-z))), name="Pr_sigmoid1")(difs)
        score = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="score")(Prs)
    else:
        if _mode!="MXFN":
            print("Invalid mode:", _mode, "- valid modes are COND, MLTM, MXFN (using default: MXFN)")
        _mode=="MXFN"
        scores = Lambda(lambda qp: qp[0] * qp[1])([qn_row, psi_row])
        score = Lambda(lambda s: K.sum(s, keepdims=True, axis=1), name="sum")(scores)
#         score = Lambda(lambda qp: K.batch_dot(qp[0], qp[1], axes=1), name="dot_prod")([qn_row, psi_row])
    #     score = Lambda(lambda z: 1.0 / (1.0 + K.exp(-z)))(score)
    
    model = Model(inputs=[qn_sel, psi_sel], outputs=score)

    if _mode=="BINQ":
        from_half = Lambda(lambda x: ((x-1)**2 * (x-0)**2)+1 )
        s_loss = from_half(psi_row)
        q_loss = from_half(qn_row)
        def custom_loss(s_loss,q_loss):
            def orig_loss(yt,yh):
                return K.binary_crossentropy(yt,yh) * s_loss * q_loss
#             return K.mean(K.square(yt-yh)) + 5000*aux_av + 1000*aux_std + aux_loss/10000
            return orig_loss
        model.compile(optimizer=optimiser, loss=custom_loss(s_loss, q_loss), metrics=["accuracy"])
        return model
    
    print("loss mode is", loss)
    if loss=="MSE":
        model.compile(optimizer=optimiser, loss="mse", metrics=["accuracy"])
    else:
        if loss!="XENT":
            print("loss mode must be MSE or XENT, not", loss," - setting to XENT.")
            loss="XENT"            
        model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["accuracy"])     
    print(model.summary())

    return model


# In[6]:


from keras.regularizers import l1
from keras.layers import Dropout, multiply, subtract, GaussianNoise, GaussianDropout, Input, Lambda, Dense
from keras import backend as K, Model
from keras.optimizers import Adam
def generate_offset_generator():
    #width, dispersal, target_EV
    
    inp = Input(shape=(3,))
    h = Dense(5, activation="relu")(inp)
    h = Dense(5, activation="relu")(h)
    h = Dense(5, activation="relu")(h)
#     h = Dense(5, activation="relu")(h)
    offset = Dense(1, activation="linear")(h)
    
    model = Model(inputs=[inp], outputs=offset)
    model.compile(optimizer=Adam(), loss="mse")
    print(model.summary())
    return model


# In[7]:


# a1 = 5/3
# a0 = 1.75
# tw = 3.5

def create_offset_generator(n_factors, min_active, max_active, sampsize=14, n_iter=20000, rasch=True):
    n_questions = int(sampsize / 0.9)
    n_students = int(sampsize / 0.9)
    gen_m = generate_offset_generator()
    inps = []
    outs = []
    
    for a in range(n_iter):

#         tw = random.uniform(0, 4)
        tw = random.uniform(0, 5)
        a0 = random.uniform(-5, 5)
        a1 = random.uniform(.5, 4)
#         print(tw,a1,"...",a0)
        
        if rasch:
            _, _, students_temp, qz_temp  = gen_rasch_run(n_factors, a0, a1, min_active, max_active, test_w = tw, n_students=n_students, n_questions=n_questions)
        else:
            _, _, students_temp, qz_temp  = gen_bayes_run(n_factors, a0, a1, min_active, max_active, test_w = tw, n_students=n_students, n_questions=n_questions)

        students2 = students_temp
        questions = qz_temp

#         print("qs set gen'd")

        probs=numpy.zeros((len(students2), len(questions)))
#         obs=numpy.zeros((len(students2), len(questions)))

        for vi in range(len(students2)):
            for mi in range(len(questions)):
                if rasch:
        #                 zmask = (qws[mi]==-10).astype(int)
    #                     print(students2[vi])
    #                     print(questions[mi])
                    deltas = students2[vi]-questions[mi]
                    prs = logistic(deltas,1,0)
    #                     print("prs=", prs)
    #                     prs = numpy.maximum(zmask,prs)
                else:
                    p = students2[vi]
                    q = questions[mi]
    #                     print("p",p)
    #                     print("q",q)
                    prs = (1-q)+(p*q)

                pr = numpy.prod(prs)
#                 print("prod pr=", pr)
    #             sz.append(vi)
    #             qz.append(mi)
#                 ob = (random.random() < pr)
                probs[vi,mi] = pr
#                 obs[vi,mi] = ob

        exp_ob = numpy.mean(probs)
        print(exp_ob)
        inps.append([tw,a1, exp_ob])
        outs.append(a0)
        
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    inps = numpy.array(inps)
    outs = numpy.array(outs)
    print(inps.shape, outs.shape)
    
    gen_m.fit(inps,outs, epochs=1000, shuffle=True, callbacks=[es], validation_split=0.1)
        
    predz = gen_m.predict(inps)
#     for i,p,o in zip(inps,predz, outs):
#         print(i, p, o)
    print("avg", numpy.mean(outs), "vs", numpy.mean(predz))
    gen_m.evaluate(inps, outs)        
    return gen_m


# In[8]:


def calc_arr_arr_err(split, real_wgts, pred_wgts, max_iter=10):
    from scipy.spatial.distance import cosine
# pred_wgts = numpy.round(pred_wgts,1)

    print(split, real_wgts.shape, pred_wgts.shape, max_iter)

    out_cols = [None] * len(real_wgts.T)
    curr_sel = None
    curr_ix = None
    n_iters = 10
    chosen = None
    curr_real_ix = None
    
    indices = range(len(real_wgts.T))

    min_total_err = math.inf
    best_dis = math.inf
    for i in range(max_iter): #len(indices)**2):
        real_used = set()
        pred_used = set()
        while len(pred_used) < len(indices):
            curr_err = math.inf
            curr_cos = math.inf
            for rix in numpy.random.permutation(indices):
                if rix in real_used:
                    continue
                real_col = real_wgts.T[rix]
                for cix in numpy.random.permutation(indices):
                    if cix in pred_used:
                        continue
                    pred_col = pred_wgts.T[cix]
                    pred_col = pred_col #* pred_q_col
                    err = numpy.mean(numpy.abs( pred_col - real_col))
                    
                    if err < curr_err:
                        curr_sel = pred_col
                        curr_err = err
                        curr_cos = 0#cosine(pred_col, real_col)
                        curr_ix = cix
                        curr_real_ix = rix
            real_used.add(curr_real_ix)
            pred_used.add(curr_ix)
            out_cols[curr_real_ix] = curr_sel
        out_col_arr = numpy.array(out_cols).T
        total_err = numpy.mean(numpy.abs( out_col_arr - real_wgts ))
        
        dis = 0
        mean_ll = numpy.mean( out_col_arr - real_wgts )
        if total_err < min_total_err:
            min_total_err = total_err
            total_q_err = numpy.mean(numpy.abs( out_col_arr[0:split] - real_wgts[0:split] ))
            total_s_err = numpy.mean(numpy.abs( out_col_arr[split:] - real_wgts[split:] ))
            best_ll = mean_ll
            chosen = out_col_arr
            best_dis = dis
    return chosen, min_total_err, total_q_err, total_s_err, mean_ll, best_dis


# In[9]:


import copy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def plot_items(pred_list, real_items, s_offset):
    if real_items.shape[1]<2:
        print("real_items is only 1 component wide .. needs to be >1 to plot on a PCA graph")
        return None
    elif real_items.shape[1]==2:
        print("2 comps, so no dim reduc")
        tx=None
    else:
        tx = PCA(n_components=2)
#         tx = TSNE(n_components=2)
    fyrst = True
    Cs = []
    C_labs = []
    pred_list = numpy.array(pred_list)
    print("pred list shape", pred_list.shape)
    print("real items shape", real_items.shape)


    fitted_pred_list = []
#     offset = numpy.median(pred_list[(pred_list>0.1)], axis=0) - numpy.median(real_items[(real_items>0.1)], axis=0)
#     real_mean = numpy.min(real_items[real_items > 0.1])
#     offset = numpy.min(pred_list[pred_list > 0.1]) - real_mean
#     print("real mean", real_mean)
#     print("offset", offset)
    
    m = len(real_items)
    cols = list(range(m))
    shuffle(cols)
    
    xmeans = numpy.zeros(m)
    ymeans = numpy.zeros(m)
    pairs = defaultdict(list)
    iter = 0
    
    cp_real = copy.copy(real_items)
    cp_real[cp_real < 1] = numpy.nan
    r_offset=numpy.nanmedian(cp_real, axis=0)
    
    itemz_2 = real_items
    n = len(real_items)
    
    for opreds in pred_list:
        preds = copy.copy(opreds) #- s_offset[iter] + r_offset
        split = 0
        
        items_chosen, min_total_err, total_q_err, total_s_err, mean_ll, best_cos_dis = calc_arr_arr_err(0, real_items, preds, max_iter=10)

        itemz_pred = items_chosen
        print(itemz_pred)
#         itemz_pred = numpy.maximum(itemz_pred,0)
        fitted_pred_list.append(itemz_pred)
        
#         itemz = real_items #- offset
#         print(numpy.min(itemz), numpy.mean(itemz), numpy.max(itemz))
#         itemz = numpy.maximum(itemz,0)
#         print(numpy.min(itemz), numpy.mean(itemz), numpy.max(itemz))
#         if itemz_2 is None:
#             itemz_2 = numpy.concatenate([real_itemz, itemz_pred], axis=0)
#         else:
        itemz_2 = numpy.concatenate([itemz_2, itemz_pred], axis=0)

    if tx:
        itemz_2 = tx.fit_transform(itemz_2)
#         itemz_2 = numpy.concatenate([itemz, itemz_pred], axis=0)
#         if fyrst:
#         itemz_2 = tx.fit_transform(itemz_2)
#             fyrst = False
#         else:
#             itemz_2 = tx.transform(itemz_2)

    from sklearn.cluster import KMeans
    iter=0
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    
    for opreds, itemz_pred in zip(pred_list, fitted_pred_list):
        n = len(itemz_pred)
        km = KMeans()
        km.fit(itemz_pred)
        cluster_labels = km.predict(itemz_pred)
        print(cluster_labels)
        
        C = []
        for l in set(cluster_labels):
            cluster = list(numpy.where(cluster_labels==l)[0])
            print("X", cluster)
            C.append(cluster)
        Cs.append(C)
        C_labs.append(cluster_labels)
                
#         NUM_COLORS = 100
#         cm = plt.get_cmap('gist_rainbow')
#         fig.gca().set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        print(type(itemz_2))
        minix=n*(iter+1)
        maxix=n*(iter+1)+n
        
        #i=0 -> 100,199
        #i=2 -> 200,299
        
        print("no pts=",n," indices=", minix, maxix)
        print("shape of itemz_2", itemz_2.shape)
        fig.gca().scatter(itemz_2[minix:maxix,0], itemz_2[minix:maxix,1], alpha=0.7, c=numpy.array(cols), cmap=plt.get_cmap('nipy_spectral'))
        j=0
        for j in range(n):
            x,xh,y,yh = itemz_2[j+(n*iter),0], itemz_2[j+(n*iter+n) ,0], itemz_2[j+(n*iter),1], itemz_2[j+(n*iter+n),1]
#             fig.gca().plot([x,xh],[y,yh],  color="#aaaaaa80")
            xmeans[j] += xh
            ymeans[j] += yh
            pairs[iter].append((xh, yh))
        iter+=1
        
    for j in range(n):
        fig.gca().annotate(j, (itemz_2[j,0], itemz_2[j,1]))
        
    fig.gca().scatter(itemz_2[0:n,0], itemz_2[0:n,1], c="b", zorder=10, alpha=0.5)
#     fig.gca().axhline(y=1e-6, linestyle="--")
#     fig.gca().axvline(x=1e-6, linestyle="--")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    xmeans = xmeans / len(pred_list)
    ymeans = ymeans / len(pred_list)
    

    if len(pred_list)>=1:
        for it in range(len(pred_list)):
            xhyh_pairs = pairs[it]
            for j,hat_pair in enumerate(xhyh_pairs):
                xh,yh = hat_pair
                x,y = itemz_2[j,0], itemz_2[j,1]
                mux = xmeans[j]
                muy = ymeans[j]
#                 fig.gca().scatter(xh, yh, alpha=0.7, c=plt.get_cmap('nipy_spectral')(cols[j]))
#                 fig.gca().scatter(mux,muy, c="#888888ff", marker="*", zorder=10)
#                 fig.gca().plot([mux,xh],[muy,yh],color="#aaaaaa80", linestyle="--")
#                 fig.gca().plot([mux,x],[muy,y],color="#888888dd", linestyle="-")
                fig.gca().plot([x,xh],[y,yh],color="#aaaaaa80", linestyle="--")
        
    plt.show()
    print("len Cs", len(Cs))
    from sklearn.metrics.cluster import adjusted_rand_score
    rands = []
    for ix in range(len(Cs)):
#         print(ix)
        for jx in range(len(Cs)):
#             print(jx)
            if ix!=jx:
#             print(Cs[ix], Cs[jx])
#                 print("VI:", ix,jx, varinfo(Cs[ix],Cs[jx]))
                a_rand = adjusted_rand_score(C_labs[ix], C_labs[jx])
                print("Rand:", a_rand)
                rands.append(a_rand)
    print("Mean rand score =", numpy.mean(rands), numpy.std(rands))

# qws = qn_table.get_weights()[0]
# qws2 = qn_table2.get_weights()[0]


# In[50]:


def generate_and_train(qz,sz,pfz, vqz,vsz,vpfz, w, n_factors, min_active, max_active, nn_mode=None, loss_mode=None):
    btm = 0
    top = math.sqrt(.1/w)
#     init= (btm,top)
#     init = math.sqrt(.5/w)
    init_s = (0,1)
    init_q = (0,1)
    
#     1-p + pq = s
#     q=0.3 : 1-p + p/3 = s
#           : 3-3p + p = s
#           : p = (3-s)/2
        
    if nn_mode=="COND":
        percompp = .5**(1/w)
        print("percompp", percompp)

        s_table =  BigTable((n_students, w), 0,1, init_hilo= percompp )#, regulariser=regularizers.l2(10e-6))
        qn_table = BigTable((n_questions, w), 0,1, init_hilo= percompp )#, regulariser=regularizers.l1(10e-6))
    elif nn_mode=="MXFN":
        init = math.sqrt(.5/w)
        print("MXFN init'n")
        print(init)
        print(init*init*w)
        s_table =  BigTable((n_students, w), -math.inf, math.inf, init_hilo= init) #, regulariser=regularizers.l2(10e-6))
        qn_table = BigTable((n_questions, w), -math.inf, math.inf, init_hilo= init) #, regulariser=regularizers.l1(10e-6))
    else:
        sp = pr_to_spread(.5, w, as_A_and_D=False)
        print("sp is ",sp)
        s_table =  BigTable((n_students, w), -math.inf, math.inf, init_hilo= 0) #, regulariser=regularizers.l2(10e-6))
        qn_table = BigTable((n_questions, w), -50, math.inf, init_hilo= -sp) #, regulariser=regularizers.l1(10e-6))        
        
    from keras.layers import Embedding
    from keras.constraints import NonNeg, MinMaxNorm
    from keras.initializers import RandomNormal, RandomUniform
    
#     wc=WeightClip(0,1)
    
#     q_gates = None #Embedding(n_questions,w, input_length=1, embeddings_initializer=RandomUniform(minval=0, maxval=1, seed=None), embeddings_constraint=wc)
#     qn_table = Embedding(n_questions,w, input_length=1, embeddings_initializer=RandomNormal(mean=6, stddev=0.3))
#     s_table = Embedding(n_students,w, input_length=1, embeddings_constraint=WeightClip(0,math.inf), embeddings_initializer=RandomNormal(mean=6, stddev=0.3))
    
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    
#     vqz=[]
    if len(vqz)>0:
        lozz="val_loss"
        val_dat= [[vqz,vsz], vpfz]
    else:
        lozz="loss"
        val_dat=None
    
    fiftiez = numpy.zeros_like(pfz) + .50
    for _ in range(1):
#         es = EarlyStopping(monitor="loss", restore_best_weights=True, patience=10)
#         m = generate_qs_model(qn_table, s_table, Adam(lr=0.001))
#         h = m.fit(x=[qz,sz], y=numpy.array(fiftiez).reshape(-1,1), batch_size=len(pfz), shuffle=True, epochs=10000, verbose=1, callbacks=[es])
#         wz = m.get_weights()
        m = generate_qs_model(qn_table, s_table, Adam(), _mode=nn_mode, loss=loss_mode)
#         m.set_weights(wz)
        predz = m.predict([vqz,vsz])
#         for vs,vq,tp,pp in zip(vsz,vqz,predz, vpfz):
#             print(vs,vq,"-",tp,pp)
        balance = numpy.mean(predz)
        print("mean pre-train pred/n:", balance)

        es = EarlyStopping(monitor=lozz, restore_best_weights=True, patience=100)
        
        h = m.fit(x=[qz,sz], y=numpy.array(pfz).reshape(-1,1), batch_size=len(pfz), shuffle=True, epochs=10000, verbose=1, callbacks=[es], validation_data=val_dat)
        
        if val_dat:
            print(m.evaluate([vqz,vsz],vpfz))
#         predz = m.predict([vqz,vsz])
#         for vs,vq,tp,pp in zip(vsz,vqz,predz, vpfz):
#             print(vs,vq,"-",tp,pp)

#     h = m.fit(x=[qz,sz], y=pfz.flatten(), batch_size=32, shuffle=True, epochs=1000, verbose=1, callbacks=[es], validation_data=[[vqz,vsz], vpfz])
    return s_table, qn_table, m, h


# In[11]:


gen_m_cache = {}

def calc_skills_coverage():
    xs=[]
    ys=[]
    ycnts=[]
    n_factors, min_active, max_active = 100,1,5
    n_students = 1
    tw=1
    a0=1
    a1=1
    max_n_qns = 150
    n_questions_list = numpy.linspace(1,max_n_qns,num=10).astype("int")

    _, _, students_temp, qz_temp  = gen_rasch_run(n_factors, a0, a1, min_active, max_active, test_w = tw, n_students=n_students, n_questions=max_n_qns)
    for n_questions in n_questions_list:
        cnt = numpy.array([False]*n_factors).astype("int")
#         plt.hist(qz_temp.flatten(), alpha=0.5)
#         plt.show()
        for q in qz_temp[0:n_questions]:
            active = (q > -40).astype("int")
#             print("a",active)
            cnt = cnt + active
            seen = numpy.clip(cnt, 0,1)
        print("c",cnt)
        print(n_questions, numpy.mean(seen), numpy.mean(cnt))
        xs.append(n_questions)
        ys.append(numpy.mean(seen))
        ycnts.append(numpy.mean(cnt))
    plt.plot(xs,ys)
    plt.plot(xs,ycnts)
    plt.show()
    
# calc_skills_coverage()

from sklearn.metrics import r2_score, mean_absolute_error


def report(n_factors, min_active, max_active, emb_w, nn_mode, loss_mode, sws_list, qws_list, model_list, real_stu_list,
           real_que_list, test_datasets, params_list):
    tot_sqerr = 0
    mean_err_list = []
    mean_std_list = []
    mean_hit_list = []

    print("*****")
    print(nn_mode, loss_mode)
    #     print("*****")
    print(len(sws_list), len(qws_list), len(model_list), len(real_stu_list), len(real_que_list), len(test_datasets),
          len(params_list))

    for sw, qw, m, stz, qnz, tt_pairs, params in zip(sws_list, qws_list, model_list, real_stu_list, real_que_list,
                                                     test_datasets, params_list):
        tw, a1, a0, trbal, vbal = params

        print("params:", n_factors, min_active, max_active, emb_w, "/", tw, a1, a0, trbal, vbal)

        err_list = []
        hit_list = []
        #     for six,qix in numpy.sort(tt_pairs, axis=0):

        true_pz = []
        pred_pz = []
        for six, qix in tt_pairs:
            #         print(six, qix)
            #     print("\n------\n")
            #     continue
            #     if False:
            tq = qnz[qix, :]
            ts = stz[six, :]
            qrow = qw[qix, :]
            srow = sw[six, :]
            #         print("raw",tq,ts)
            #         print("dif",ts-tq)
            #         print(numpy.prod(logistic(ts-tq,1,0)))
            if rasch:
                true_p = numpy.prod(logistic(ts - tq, 1, 0))
            else:
                true_p = numpy.prod((1 - tq) + (ts * tq))
            pred_p = m.predict([[qix], [six]])
            true_pz.append(float(true_p))
            pred_pz.append(float(pred_p))
            #         pred_p = random.random()

            mae = numpy.abs(true_p - pred_p)
            print(true_p, float(pred_p), "err:", float(mae))

            err_list.append(mae)
            good_guess = int(numpy.round(true_p)) == int(numpy.round(pred_p))
            hit_list.append(int(good_guess))
        #         sqerr = numpy.power(true_p - pred_p, 2)

        #         print(six, qix, ":", srow, qrow)
        #         print("-->", pred_p, true_p, " ... ", good_guess)

        print("R2 = ", r2_score(true_pz, pred_pz))
        print("MAE = ", mean_absolute_error(true_pz, pred_pz))
        numpy.set_printoptions(precision=3)
        #     print("Mean sq err {}:".format(qrow.shape), numpy.sqrt(numpy.mean(err_list)))
        mean_err_list.append(numpy.mean(err_list))
        mean_std_list.append(numpy.std(err_list))
        mean_hit_list.append(numpy.mean(hit_list))
    #     print(sum(hit_list), len(hit_list), sum(hit_list)/len(hit_list))

    # print(mean_err_list)
    # print(mean_std_list)
    # print(mean_hit_list)
    # print(params_list)
    print(len(stz), "x", len(qnz))
    #     for e,s,acc,params in zip(mean_err_list, mean_std_list, mean_hit_list, params_list):
    #         print("acc=",acc)
    #         print("mae=",e,"sig=",s)
    #         print(params)
    #     print("aggregated:")
    print(numpy.median(mean_hit_list), numpy.std(mean_hit_list), "/", numpy.median(mean_err_list),
          numpy.median(mean_std_list))


# In[ ]:


#tw should be ~U[0.5, 3.5]
#sw should be ~N[0, sd] with sd ~U[1, 3.5]
#a0 should be ~U[-0.5, 1]
#missing proportion should be ~U[0, 0.3]

from tensorflow import set_random_seed

explore_mode = True

reportz=[]

# factors_master = [(10,1,5)]
factors_master = [(10,10,10)]
w_list = [1,10]
factors_list = [ m+(w,) for m in factors_master for w in w_list ]

# nn_modes = ["MXFN","COND","MLTM"]
nn_modes = ["MLTM"]
loss_modes = ["XENT"]
sq_nums = [(1000, 150)]
# student_staminas = [0.01, 0.1, 0.5, 0.75, 1.0]

def stitch_n_split(pairs, probs):
#     _pfz = numpy.array([int((random.random() < probs[vi,mi])) for (vi,mi) in pairs])
    _pfz = numpy.array([int(probs[vi,mi]>=0.5) for (vi,mi) in pairs])
    _sz = [p[0] for p in pairs]
    _qz = [p[1] for p in pairs]
    return _pfz, _sz, _qz

n_runs = 1

for (n_students, n_questions) in sq_nums:
    for nn_mode in nn_modes:
        for loss_mode in loss_modes:
            for (n_factors, min_active, max_active, emb_w) in factors_list:
            
                model_list=[]
                rasch=True

                questions=None

                tup = (n_factors, min_active, max_active) 
                if tup in gen_m_cache:
                    gen_m = gen_m_cache[tup]
                else:
                    gen_m = create_offset_generator(n_factors, min_active, max_active, sampsize=14, n_iter=20000)
                    gen_m_cache[tup] = gen_m

                qws_list = []
                sws_list = []
                tr_list = []
                params_list = []
                # questions=None
                real_stu_list=[]
                real_que_list=[]
                test_datasets=[]
#                 qn_av = None
#                 qn_std = None

                pred_list = []
                
                set_random_seed(666)
                numpy.random.seed(666)
                for a in range(n_runs):
                    found = False
                    while not found:
                        tw = random.uniform(0.5, 3.5)
                        a1 = random.uniform(1, 3.5)
                        a0 = gen_m.predict(numpy.array([[tw,a1,0.5]]).reshape(1,-1))
                        
                        if rasch:
                            _, _, students_temp, qz_temp  = gen_rasch_run(n_factors, a0, a1, min_active, max_active, test_w = tw, n_students=n_students, n_questions=n_questions)
                        else:
                            _, _, students_temp, qz_temp  = gen_bayes_run(n_factors, a0, a1, min_active, max_active, test_w = tw, n_students=n_students, n_questions=n_questions)


                        students2 = students_temp

                    #     if questions is None:
                        questions = qz_temp

    #                         qn_av = numpy.mean(questions, axis=0)
    #                         qn_std = numpy.std(questions, axis=0)

                        if explore_mode:
                            plot_items([], questions, None)

                            print("~ ~ ~ ~~ ATTEMPT",a, a0)
                            bin_spread = lambda x: max(1,int(abs(2*(numpy.max(x)-numpy.min(x)))))

                            plt.hist(students2.flatten(), alpha=0.5, bins=bin_spread(students2))
                            plt.hist(questions.flatten(), alpha=0.5, bins=bin_spread(questions))
                            plt.show()

                    #     (sz2,qz2,pfz2), (vsz2,vqz2,vpfz2), (tsz2,tqz2,tpfz2), obs2, probs2 = tvt_split(students2, questions, split_mode=1)
                    #     tr_list.append(((sz2,qz2,pfz2), (vsz2,vqz2,vpfz2), (tsz2,tqz2,tpfz2)))

                        sz=[]
                        qz=[]
                        pfz=[]
                        probs=numpy.zeros((len(students2), len(questions)))
                        obs=numpy.zeros((len(students2), len(questions)))

                        all_pairs = []
                        for vi in range(len(students2)):
                            for mi in range(len(questions)):
                                for _ in range(1):

                                    if rasch:
                            #                 zmask = (qws[mi]==-10).astype(int)
                    #                     print(students2[vi])
                    #                     print(questions[mi])
                                        deltas = students2[vi]-questions[mi]
                                        prs = logistic(deltas,1,0)
                    #                     print("prs=", prs)
                    #                     prs = numpy.maximum(zmask,prs)
                                    else:
                                        p = students2[vi]
                                        q = questions[mi]
                    #                     print("p",p)
                    #                     print("q",q)
                                        prs = (1-q)+(p*q)

                                    pr = numpy.prod(prs)
                    #                 print("prod pr=", pr)
    #                                 sz.append(vi)
    #                                 qz.append(mi)
                                    all_pairs.append((vi,mi))
    #                                 ob = (random.random() < pr)
                                    probs[vi,mi] = pr
                    #                 obs[vi.mi] = ob                
    #                                 pfz.append(pr)

    #                     all_pairs = list(zip(sz,qz))
                        shuffle(all_pairs)

                    #     divvy = len(all_pairs)//10
                    #     print("divvy",divvy)
                        divvy = min(1000, len(all_pairs)//20)

                        from sklearn.model_selection import train_test_split
    #                     dummyz = numpy.zeros(len(all_pairs))
                        tr_pairs, tt_pairs = train_test_split(all_pairs, test_size=divvy)
                        tr_pairs, v_pairs = train_test_split(tr_pairs, test_size=divvy)
                        
                        for pa in tr_pairs:
#                             print(pa)
                            if pa in tt_pairs:
                                print("TR IN TT")
                                raise Exception
                            if pa in v_pairs:
                                print("TR IN V")
                                raise Exception

                        pfz, sz, qz = stitch_n_split(tr_pairs, probs)
                        vpfz, vsz, vqz = stitch_n_split(v_pairs, probs)
                    #     vpfz, vsz, vqz = [],[],[]

                        print("lens of pfz and vpfz", len(pfz), len(vpfz))

                        if explore_mode:
                            plt.hist(probs.flatten(), alpha=0.5)
                            plt.title("probs")
                            plt.show()

                            plt.hist(numpy.array(pfz).flatten(), alpha=0.5)
                            plt.title("pfz")
                            plt.show()

                        print(tw, a1, a0)
                        mn = numpy.mean(pfz)
                        print(mn, numpy.mean(vpfz))
                        if mn >=0.45 and mn <=0.55:
                            print("FOUND ELIGIBLE SPREAD")
                            found=True
                            
                    real_stu_list.append(students2)
                    real_que_list.append(questions)
                    test_datasets.append(tt_pairs)
                    params_list.append((tw,a1,a0,numpy.mean(pfz), numpy.mean(vpfz)))
                #     if numpy.mean(pfz) <0.4 or numpy.mean(pfz)>0.6:
                #         continue

                # for runix in range(n_runs):
                #     (sz2,qz2,pfz2), (vsz2,vqz2,vpfz2), (tsz2,tqz2,tpfz2) = tr_list[runix]
                    obs_are_binary = numpy.array_equal(numpy.array(pfz).flatten(), numpy.array(pfz).flatten().astype(bool))
                    print("binary obs?", obs_are_binary)

                    print("callio:")
                    print(len(qz),len(sz),len(pfz))
                    print(len(vqz),len(vsz),len(vpfz), emb_w)
#                     nn_mode = "MLTM"
#                     loss_mode = "XENT"
                    print("nn_mode", nn_mode)
                    s_table2, qn_table2, m2, h2 = generate_and_train(qz,sz,pfz, vqz,vsz,vpfz, emb_w, n_factors, min_active, max_active, nn_mode=nn_mode, loss_mode=loss_mode)
#                     qws2= copy.copy(qn_table2.get_weights()[0])
#                     sws2= copy.copy(s_table2.get_weights()[0])
                    qws2= qn_table2.get_weights()[0]
                    sws2= s_table2.get_weights()[0]
                    
                    pred_probs = m2.predict([qz, sz])
#                     print(pred_probs)
                    pred_list.append(pred_probs)
                    model_list.append(m2)

                #     qg = q_gates.get_weights()[0]
                #     qg_list.append(qg)
                #     if qn_av is None:
                #         qn_av = numpy.mean(qws2)

                    sws_list.append(sws2)
                    qws_list.append(qws2)
                tup = (n_factors, min_active, max_active, emb_w, nn_mode, loss_mode, sws_list, qws_list, model_list, real_stu_list, real_que_list, test_datasets, params_list)
#                 reportz.append(zlib.compress(pickle.dumps(tup)))
                reportz.append(tup)
    


# In[56]:


for tup in reportz:
#     tup = pickle.loads(zlib.decompress(tup_cmp))
    report(*tup)
    

