from importlib import reload

from collections import defaultdict

import numpy

import random
from matplotlib import pyplot as plt
from random import shuffle, choice, randint

import math
import keras
import tensorflow

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


from sklearn.svm import SVR


def calc_probs_from_embs(students,questions):
    students2 = numpy.repeat(students, len(questions), axis=0)
    questions2 = numpy.tile(questions, (len(students),1))
    zmask = numpy.isclose(questions2,-10).astype(int)
    diffs = students2-questions2
    prs = 1.0/(1.0+ numpy.exp(-diffs))
    prs = numpy.maximum(zmask,prs)
    probs2 = numpy.prod(prs, axis=1).reshape(len(students), len(questions))
    return probs2


# In[2]:


from keras.regularizers import l1
from keras.layers import Reshape, Dense, Dropout, multiply, subtract, GaussianNoise, GaussianDropout, Input, Lambda, Embedding, concatenate, Flatten
from keras import backend as K, Model
def generate_qs_model(qn_table, psi_table, optimiser, _mode="MXFN", loss="MSE"):
    
    psi_sel = Input(shape=(1,), name="psi_select", dtype="int32")
    qn_sel = Input(shape=(1,), name="q_select", dtype="int32")

#     print(qn_table, psi_table, psi_sel, qn_sel)
#     print("psi_sel shape", psi_sel.shape)

    psi_table.trainable=True
    qn_table.trainable=True
    
    qn_row = qn_table(qn_sel)
    psi_row = psi_table(psi_sel)
    
    row_w = qn_table.kernel.shape[1]

    print("Mode is", _mode)
    if _mode=="DEEP":
        qn_row = Embedding(qn_table.kernel.shape[0] , row_w, input_length=1)(qn_sel)
        psi_row = Embedding(psi_table.kernel.shape[0], row_w, input_length=1)(psi_sel)

        qn_table = None
        psi_table = None
        
#         qn_row = Reshape((row_w, ))(qn_row)
#         psi_row = Reshape((row_w, ))(psi_row)
        qn_row = Flatten()(qn_row)
        psi_row = Flatten()(psi_row)
    
        loss = "XENT"
        
        difs = subtract([psi_row, qn_row], name="difs")
#         difs = Dropout(0.05)(difs)
        Prs = Lambda(lambda z: (K.exp(z) / (1.0 + K.exp(z))), name="Pr_sigmoid1")(difs)
#         Prs = Dropout(0.05)(Prs)
#         hazardrat = Lambda(lambda ps: K.prod(ps), name="hazard_tse")(Prs)
#         h = Dropout(0.01)(Prs)

#         h = concatenate([psi_row, qn_row])
#         h = Dense(2*row_w, activation="relu")(h)
#         h = Dropout(0.1)(h)
#         h = Dense(100, activation="relu")(h)
# #         h = Dense(5, activation="relu")(h)
#         h = Dropout(0.1)(h)
#         hazard = Dense(1, activation="sigmoid", name="hazard")(h)
    
    
        h = concatenate([psi_row, qn_row])
#         h = Dense(128, activation="relu")(h)
#         h = Dropout(0.01)(h)
#         h = Dense(64, activation="relu")(h)
#         h = Dropout(0.01)(h)
#         h = Dense(10, activation="relu")(h)
#         h = Dropout(0.01)(h)
        h = Dense(5, activation="relu")(h)
#         h = Dropout(0.01)(h)
        hazard = Dense(1, activation="sigmoid", name="hazard")(h)
    
#         psi_row = Dense(32, activation="relu")(psi_row)
#         psi_row = Dense(32, activation="relu")(psi_row)
#         psi_row = Dense(32, activation="relu")(psi_row)
        
#         qn_row = Dense(32, activation="relu")(qn_row)
#         qn_row = Dense(32, activation="relu")(qn_row)
#         qn_row = Dense(32, activation="relu")(qn_row)
        
#         psi_row = Dense(32, activation="relu")(psi_row)
#         qn_row = Dense(32, activation="relu")(qn_row)
#         h = concatenate([psi_row, qn_row, difs])
#         psi_row = Dense(13, activation="relu")(psi_row)
#         qn_row = Dense(13, activation="relu")(qn_row)
#         h = concatenate([psi_row, qn_row])
#         h = Dense(64, activation="relu")(difs)
#         h = Dense(50, activation="relu")(h)
#         h = concatenate([h, difs])

#         h = Dense(64, activation="relu")(h)

#         h = Dense(100, activation="relu")(h)
#         hazard = Dense(10, activation="relu")(h)

        hazard = Dense(10, activation="relu")(hazard)
#         hazard = Dropout(0.1)(hazard)
        hazard = Dense(10, activation="relu")(hazard)
#         hazard = Dropout(0.1)(hazard)
        score = Dense(5, activation="softmax")(hazard)

        
#         difs = subtract([psi_row, qn_row])
#         h = Lambda(lambda z: (2*(1.0 / (1.0 + K.exp(-z))))-0.5), name="Pr_sigmoid1")(difs)
#         h = Dense(64, activation="sigmoid")(h)
#         h = difs
#         h = Dense(10, activation="relu")(h)
#         h = Dense(50, activation="relu")(h)
#         h = Dense(10, activation="relu")(h)
#         h = Dense(3, activation="relu")(h)
#         h = Dense(3, activation="relu")(h)
#         h = Dense(3, activation="relu")(h)
#         hazard = Dense(1, activation="sigmoid")(h)
#         score = Dense(25, activation="softmax")(h)
        
#         outz=[]
#         max_failures = 4
#         for n in range(max_failures):
#             print("setting o for ",n)
#             mid = Lambda(lambda p: (1-p)**n * p)
#             o = mid(hazard)
#             outz.append(o)
# #         fin = Lambda(lambda p: (1-p)**(max_failures))(score)
#         fin = Lambda(lambda pz: 1.0 - K.sum(pz, axis=-1, keepdims=True))(concatenate(outz))
#         outz.append(fin)
#         score = concatenate(outz)

        
    elif _mode=="COND":
        Prs = Lambda(lambda qs: (1 - qs[0])+(qs[0]*qs[1]), name="Prs")([qn_row, psi_row])
        score = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="score")(Prs)
#     elif _mode=="BINQ":
#         Prs = Lambda(lambda qs: (1 - qs[0])+(qs[0]*qs[1]), name="Prs")([qn_row, psi_row])
#         score = Lambda(lambda ps: K.prod(ps, axis=1, keepdims=True), name="score")(Prs)
    elif _mode=="MLTM":
        klip = Lambda(lambda qk: K.clip(qk,-10, -9)+10)
        q_masque = klip(qn_row)
        difs = subtract([psi_row, qn_row])
        Prs = Lambda(lambda z: (1.0 / (1.0 + K.exp(-z))), name="Pr_sigmoid1")(difs)
        Prs = Lambda(lambda ps_q:  ps_q[0]*ps_q[1] + (1-ps_q[1]) ) ([Prs, q_masque])
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

#     if _mode=="BINQ":
#         from_half = Lambda(lambda x: 1+K.sum((0.25-(x-0.5)**2)) )
#         s_loss = from_half(psi_table.kernel)
#         q_loss = from_half(qn_table.kernel)
#         def custom_loss(s_loss,q_loss):
#             def orig_loss(yt,yh):
#                 return K.binary_crossentropy(yt,yh) * s_loss * q_loss
# #             return K.mean(K.square(yt-yh)) + 5000*aux_av + 1000*aux_std + aux_loss/10000
#             return orig_loss
#         model.compile(optimizer=optimiser, loss=custom_loss(s_loss, q_loss), metrics=["accuracy"])
#         return model
    
    print("loss mode is", loss)
    if loss=="MSE":
        model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    elif loss=="XENT":
        model.compile(optimizer=optimiser, loss="categorical_crossentropy", metrics=["mean_absolute_error"])     
    else:
        model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["mean_absolute_error"])     
    print(model.summary())

    return model


# In[3]:


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


# In[4]:


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


# In[5]:


from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
def generate_and_train(n_students, n_questions, qz,sz,pfz, vqz,vsz,vpfz, w, n_factors, min_active, max_active, nn_mode=None, loss_mode=None, class_weights=None):
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
    elif nn_mode=="MLTM":
        sp = pr_to_spread(.5, w, as_A_and_D=False)
        print("sp is ",sp)
        s_table =  BigTable((n_students, w), -math.inf, math.inf, init_hilo= 0) #, regulariser=regularizers.l2(10e-6))
        qn_table = BigTable((n_questions, w), -math.inf, math.inf, init_hilo= -sp) #, regulariser=regularizers.l1(10e-6))        
    else:
        s_table =  BigTable((n_students, w), -math.inf, math.inf, init_hilo= 0) #, regulariser=regularizers.l2(10e-6))
        qn_table = BigTable((n_questions, w), -math.inf, math.inf, init_hilo= 0) #, regulariser=regularizers.l1(10e-6))        
                
    from keras.layers import Embedding
    from keras.constraints import NonNeg, MinMaxNorm
    from keras.initializers import RandomNormal, RandomUniform
    
#     wc=WeightClip(0,1)
    
#     q_gates = None #Embedding(n_questions,w, input_length=1, embeddings_initializer=RandomUniform(minval=0, maxval=1, seed=None), embeddings_constraint=wc)
#     qn_table = Embedding(n_questions,w, input_length=1, embeddings_initializer=RandomNormal(mean=6, stddev=0.3))
#     s_table = Embedding(n_students,w, input_length=1, embeddings_constraint=WeightClip(0,math.inf), embeddings_initializer=RandomNormal(mean=6, stddev=0.3))
    
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, LambdaCallback
    
#     vqz=[]
    if len(vqz)>0:
        lozz="val_mean_absolute_error"
        val_dat= [[vqz,vsz], vpfz]
    else:
#     if True:
        lozz="mean_absolute_error"
        val_dat=None
    
    fiftiez = numpy.zeros_like(pfz) + .50
    for _ in range(1):
#         es = EarlyStopping(monitor="loss", restore_best_weights=True, patience=10)
#         m = generate_qs_model(qn_table, s_table, Adam(lr=0.001))
#         h = m.fit(x=[qz,sz], y=numpy.array(fiftiez).reshape(-1,1), batch_size=len(pfz), shuffle=True, epochs=10000, verbose=1, callbacks=[es])
#         wz = m.get_weights()
        m = generate_qs_model(qn_table, s_table, Adam(), _mode=nn_mode, loss=loss_mode)
#         m.set_weights(wz)
        tr_predz = (m.predict([qz,sz])[:,0] > 0.5)
#         for vs,vq,tp,pp in zip(vsz,vqz,predz, vpfz):
#             print(vs,vq,"-",tp,pp)
        print("PRE-TR AVG  = ", numpy.mean(tr_predz))
    
        if len(vqz)>0:
            v_predz  = (m.predict([vqz,vsz])[:,0] > 0.5)
            print("PRE-TR VAVG = ", numpy.mean(v_predz))

        es = EarlyStopping(monitor=lozz, restore_best_weights=True, patience=10)
        
        

        
        intermediate_layer_model = Model(inputs=m.input,
                                outputs=m.get_layer("hazard").output)
        intermediate_output = intermediate_layer_model.predict([qz,sz])
        
        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: 
                                       print(numpy.min(intermediate_layer_model.predict([qz,sz])),
                                             numpy.max(intermediate_layer_model.predict([qz,sz]))))

        
        _bs = len(pfz)
#         _bs = 1000
        for _ in range(1):
            h = m.fit(x=[qz,sz], y=pfz, batch_size=_bs, class_weight=class_weights, shuffle=True, epochs=10000, verbose=1, callbacks=[es], validation_data=val_dat)
        tr_predz = m.predict([qz,sz])  
        if len(vqz)>0:
            v_predz  = m.predict([vqz,vsz])

#         print("TR AVG = ", numpy.mean(tr_predz))
#         print("TR R2  = ", r2_score(pfz, tr_predz))
#         print("TR MAE = ", mean_absolute_error(pfz, tr_predz))
#         print("TR ACC = ", accuracy_score((pfz>0.5), (tr_predz>0.5)))
#         print("TR AGT = ", accuracy_score([random.random() < p for p in pfz], [random.random() < p for p in tr_predz]))
        
#         if val_dat:
#             print("VA AVG = ", numpy.mean(v_predz))
#             print("VA R2  = ", r2_score(vpfz, v_predz))
#             print("VA MAE = ", mean_absolute_error(vpfz, v_predz))
#             print("VA ACC = ", accuracy_score((vpfz>0.5), (v_predz>0.5)))
#             print("VA AGT = ", accuracy_score([random.random() < p for p in vpfz], [random.random() < p for p in v_predz]))

#             for ent,hat in zip(vpfz, v_predz):
#                 print(ent)
#                 print(hat)
#                 print(numpy.sum(hat))
#                 print("~~~~")

    from sklearn.metrics import classification_report
    print(classification_report((pfz>0.5), (tr_predz>0.5)))
    if len(vqz)>0:
        print(classification_report((vpfz>0.5), (v_predz>0.5)))
            
#     h = m.fit(x=[qz,sz], y=pfz.flatten(), batch_size=32, shuffle=True, epochs=1000, verbose=1, callbacks=[es], validation_data=[[vqz,vsz], vpfz])
    return s_table, qn_table, m, h


# In[6]:


def stitch_n_split(_pairs, sts, qns, realise=True, rpt=False):
    def calc_probs(s,q):
        zmask = numpy.isclose(q,-10).astype(int)
        diff = s-q
        prs = 1.0/(1.0+ numpy.exp(-diff))
        prs = numpy.maximum(zmask,prs)
        pr = numpy.prod(prs, axis=1).reshape(len(q))
        return pr

    out_w = 5
    max_fails = out_w -1
    if realise:
        if rpt:
            _counts = defaultdict(int)
            _matches=[]
            _pfz=[]
            _sz, _qz = [],[]
            cache = defaultdict(list)
            for (vi,mi) in _pairs:
                cache[vi].append(mi)
            
            for vi in cache:
                s = sts[vi]
                prs = calc_probs(s, [qns[k] for k in cache[vi]])  
#                 print("shape of prs", prs.shape)
                for mix,mi in enumerate(cache[vi]):
#                 pr = _probs[vi,mi]
                    pr = prs[mix]
                    rnd = random.random()
                    i=0
                    while rnd > pr and i<max_fails:
                        i += 1
                        rnd = random.random()

                    zs = numpy.zeros(max_fails+1)
                    zs[i] = 1
                    _counts[i] += 1

                    _pfz.append(zs)
    #                 _pfz.append(i)
                    _sz.append(vi)
                    _qz.append(mi)
            print("probs calced")
        else:
            _prob_list =  numpy.array([calc_prob(sts[vi],qns[mi]) for (vi,mi) in _pairs])
            _pfz = (numpy.random.random(len(_prob_list)) < _prob_list).astype(int)
    #     _pfz = (0.5 < _prob_list).astype(int)
            _matches = ( numpy.round(_prob_list) == _pfz).astype(int)
        print("realisation complete")
    else:
        _prob_list =  numpy.array([calc_prob(sts[vi],qns[mi]) for (vi,mi) in _pairs])
        _pfz = _prob_list
        _matches = numpy.ones_like(_prob_list)
        _sz = [p[0] for p in _pairs]
        _qz = [p[1] for p in _pairs]

#     print(_pfz)
#     _pfz = numpy.array([probs[vi,mi] for (vi,mi) in _pairs])

#     print(_matches)
#     print(numpy.sum(_matches), "correctly labelled out of", len(_matches), "%=", numpy.sum(_matches)/len(_matches))
    if not rpt:
        _sz = [p[0] for p in _pairs]
        _qz = [p[1] for p in _pairs]
    print("sns complete")
    
    one_c = _counts[1]
    for k in _counts:
        c = _counts[k]
        _counts[k] = one_c / c
    
    return numpy.array(_pfz), _sz, _qz, _counts


# In[7]:


gen_m_cache = pickle.load(open("generators.p", "rb"))


# In[8]:


# gen_m_cache = {}


# In[9]:


from sklearn.metrics import r2_score, mean_absolute_error

def report(n_factors, min_active, max_active, emb_w, nn_mode, loss_mode, sws_list, qws_list, model_list, real_stu_list, real_que_list, test_datasets, params_list, spars_list, compare=False):
    
    tot_sqerr = 0
    mean_err_list = []
    mean_std_list = []
    mean_hit_list = []
    
    print("*****")
    print(nn_mode, loss_mode)
#     print("*****")
    print(len(sws_list), len(qws_list), len(model_list), len(real_stu_list), len(real_que_list), len(test_datasets), len(params_list))
    
    for sw,qw,m,stz,qnz,tt_pairs, params, spars in zip(sws_list, qws_list, model_list, real_stu_list, real_que_list, test_datasets, params_list, spars_list):
        tw,a1,a0,trbal,vbal,agt = params
        
        print("params:", n_factors, min_active, max_active, emb_w, "/", tw,a1,a0, "(", trbal,vbal,agt,") [",spars,"]")
        
        err_list = []
        true_err_list = []
        hit_list = []
    #     for six,qix in numpy.sort(tt_pairs, axis=0):
    
        true_pz = []
        pred_pz = []
        for six, qix in tt_pairs:
    #         print(six, qix)
    #     print("\n------\n")
    #     continue
    #     if False:
            tq = qnz[qix,:]
            ts = stz[six,:]
            qrow = qw[qix, :]
            srow = sw[six, :]
#             print(qrow)
    #         print("raw",tq,ts)
    #         print("dif",ts-tq)
    #         print(numpy.prod(logistic(ts-tq,1,0)))
#             if rasch:
            true_p = float(calc_probs_from_embs(ts.reshape(1,-1),tq.reshape(1,-1)))
#                 dif = ts-tq
#                 true_ps = 1.0 / (1.0 + numpy.exp(-dif))
#                 true_p = numpy.prod(true_ps)
#             else:
#                 true_p = numpy.prod((1-tq)+(ts*tq))
            pred_p = m.predict([[qix],[six]]).flatten()[0]
    
            if compare:
                print(six,qix, ":", true_p, pred_p)
    
            true_pz.append(float(true_p))
            pred_pz.append(float(pred_p))
    #         pred_p = random.random()
    
            mae = numpy.abs(true_p - pred_p)
#             print(true_p, float(pred_p), "err:", float(mae))

            err = true_p - pred_p

            true_err_list.append(err)
            err_list.append(mae)
            good_guess = int(numpy.round(true_p))==int(numpy.round(pred_p))
            hit_list.append(int(good_guess))
    #         sqerr = numpy.power(true_p - pred_p, 2)

#             print(six, qix, ":", srow, qrow)
#             print("-->", pred_p, true_p, " ... ", good_guess)

        print("R2 = ", r2_score(true_pz, pred_pz))
        print("MAE = ", mean_absolute_error(true_pz, pred_pz))
        numpy.set_printoptions(precision=3)
    #     print("Mean sq err {}:".format(qrow.shape), numpy.sqrt(numpy.mean(err_list)))
    
        plt.hist(true_pz, alpha=0.5)
        plt.hist(pred_pz, alpha=0.5)
        plt.show()
        
        plt.hist(numpy.array(true_err_list).flatten(), alpha=0.5)
        plt.show()
        
        mean_err_list.append(numpy.mean(err_list))
        mean_std_list.append(numpy.std(err_list))
        mean_hit_list.append(numpy.mean(hit_list))
    #     print(sum(hit_list), len(hit_list), sum(hit_list)/len(hit_list))

    # print(mean_err_list)
    # print(mean_std_list)
    # print(mean_hit_list)
    # print(params_list)
    print(len(stz),"x",len(qnz))
#     for e,s,acc,params in zip(mean_err_list, mean_std_list, mean_hit_list, params_list):
#         print("acc=",acc)
#         print("mae=",e,"sig=",s)
#         print(params)
#     print("aggregated:")
    print(numpy.median(mean_hit_list), numpy.std(mean_hit_list), "/", numpy.median(mean_err_list), numpy.median(mean_std_list))
    print(numpy.median(mean_err_list), numpy.mean(mean_err_list))
# report(n_factors, min_active, max_active, emb_w, nn_mode, loss_mode, sws_list, qws_list, model_list, real_stu_list, real_que_list, test_datasets, params_list)


# In[10]:


data_cache = {}


# In[11]:


##### tw should be ~U[0.5, 3.5]
#sw should be ~N[0, sd] with sd ~U[1, 3.5]
#a0 should be ~U[-0.5, 1]
#missing proportion should be ~U[0, 0.3]

# from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split

explore_mode = False

reportz=[]

# factors_master = [(10,1,5)]
factors_master = [(100,1,5)]
w_list = [100]
factors_list = [ m+(w,) for m in factors_master for w in w_list ]

# nn_modes = ["MLTM","COND","MXFN"]
nn_modes = ["DEEP"]
loss_modes = ["MSE"]
# sq_nums = [(int(1000*(1.4)**3), int(150*(1.4)**3))]
# sq_nums = [(int(200*(1.41)**4), int(200*(1.41)**4))]
sq_nums = [(10000, 1000)]

# student_staminas = [0.01, 0.1, 0.5, 0.75, 1.0]

spars_list = [1] # [0.01, 0.05, 0.25, 0.5, 0.75, 1.0]

data_to_run = [0]#,5]
bal = .5


for (n_students, n_questions) in sq_nums:
    print("{} students, {} questions".format(n_students, n_questions))
    for nn_mode in nn_modes:
        for loss_mode in loss_modes:
            for (n_factors, min_active, max_active, emb_w) in factors_list:
                for spars in spars_list:
                    
                    tup = ((n_factors, min_active, max_active), (10000, 70))
                    if tup in gen_m_cache:
                        (gen_m, history, best_dims, best_mse) = gen_m_cache[tup]
                    else:
                        print(gen_m_cache.keys())
                        raise Exception("Genny not found for",tup)

#                     set_seed(666)
                    numpy.random.seed(666)

                    pred_list  = []
                    model_list = []
                    sparss     = []
                    sws_list   = []
                    qws_list   = []
                    real_stu_list = []
                    real_que_list = []
                    params_list   = []
                    test_datasets = []
                    for a in data_to_run:

                        (tw,a0,a1, students_temp, qz_temp) = pickle.load(open("./synth_data/MLTM_10000_1000_(100_1_5)_{}.p".format(a), "rb"))
                        print("loaded dataset",a,":", a0,a1,tw)
                                
                        snan = numpy.isnan(numpy.sum(students_temp))
                        qnan = numpy.isnan(numpy.sum(qz_temp))
                        print(snan, qnan)
                            
#                         students2 = students_temp.astype(int)[0:1000]
#                         questions = qz_temp.astype(int)
                        
                        students2 = students_temp[0:2000]
                        questions = qz_temp

                        if explore_mode:
                            plot_items([], questions, None)

                            print("~ ~ ~ ~~ ATTEMPT",a, a0)
                            bin_spread = lambda x: max(1,int(abs(2*(numpy.max(x)-numpy.min(x)))))

                            plt.hist(students2.flatten(), alpha=0.5, bins=bin_spread(students2))
                            plt.hist(questions.flatten(), alpha=0.5, bins=bin_spread(questions))
                            plt.show()

                        tr_pairs = []
                        v_pairs = []
                        tt_pairs = []
                        slist = list(range(len(students2)))
                        random.seed(666)
                        shuffle(slist)
                        for vi in slist:
                            qlist= list(range(len(questions)))
                            shuffle(qlist)
                            first = True
                            for mi in qlist:
                                if first:
                                    tt_pairs.append((vi,mi))
                                    first = False
                                else:
                                    tr_pairs.append((vi,mi))

                        print("splitting")
                        realise = True
                        if spars < 1:
                            tr_pairs, _ = train_test_split(tr_pairs, train_size=spars)
                        tr_pairs, v_pairs = train_test_split(tr_pairs, test_size=0.1)
                        print("splut")

                        pfz, sz, qz, _ = stitch_n_split(tr_pairs, students2, questions, realise=realise, rpt=True)
                        vpfz, vsz, vqz, _ = stitch_n_split(v_pairs, students2, questions, realise=realise, rpt=True)

                        print("SS done")


                        print("Sparsity",spars,"lens of pfz and vpfz, tt_pairs", len(pfz), len(vpfz), len(tt_pairs))

                        if explore_mode:
                            plt.hist(numpy.array(pfz).flatten(), alpha=0.5)
                            plt.title("pfz")
                            plt.show()


                        class_weights = None
    #                     data_cache[tup] = (students_temp, qz_temp, (pfz, sz, qz),(vpfz, vsz, vqz), class_weights)


        #                 print("mean pers is", numpy.mean(perseverance))
        #                 perseverance_list.append(perseverance)
                        real_stu_list.append(students2)
                        real_que_list.append(questions)
                        test_datasets.append(tt_pairs)
                        agt = None
                        params_list.append((tw,a1,a0,numpy.mean(pfz), numpy.mean(vpfz), agt))
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
                        s_table2, qn_table2, m2, h2 = generate_and_train(n_students, n_questions, qz,sz,pfz, vqz,vsz,vpfz, emb_w, n_factors, min_active, max_active, nn_mode=nn_mode, loss_mode=loss_mode, class_weights=None)
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

                        sparss.append(spars)
                        sws_list.append(sws2)
                        qws_list.append(qws2)
                        tup = (n_factors, min_active, max_active, emb_w, nn_mode, loss_mode, sws_list, qws_list, model_list, real_stu_list, real_que_list, test_datasets, params_list, sparss)
        #                 reportz.append(zlib.compress(pickle.dumps(tup)))
        #                 print(perseverance_list)

                    reportz.append(tup)
print("finished")


# In[ ]:


print(len(reportz))
for tup in reportz:
#     tup = pickle.loads(zlib.decompress(tup_cmp))/
    report(*tup, compare = True)
    

