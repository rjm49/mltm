from random import choice, randint, random, uniform
import math
from math import exp


def generate_sequence(n_timesteps):
    return [random() for _ in range(n_timesteps)]

def generate_student_name(n_names=2, maxl=5):
    # cons = ["B","C","CH","CK","D","F","G","GH","H","J","K","L","LL","M","N","P","PH","QU","R","RR","S","SH","SS","T","TH","V","W","WH","X","Y","Z"]
    cons = ["B","C","CH","D","F","G","H","J","K","L","M","N","P","PH","R","S","T","TH","V","W","Y"]
    verbs = ["A","E","I","O","U","Y"]#,"OO","OU","UI","AE","EI","IE","EE"]
    N = ""
    for n in range(n_names):
        L = randint(3, max(3,maxl))
        i=0
        while i < L:
            v = choice(verbs)
            N = N + ( choice(cons) if (i%2==0) else choice(verbs) )
            i+=1
        N = N + " "
    return N

import classes
def create_qs(n_qs, beta_min, beta_max, nt, min_active, max_active):
    master_qs = [classes.Question(qix, beta_min,beta_max, nt, min_active, max_active) for qix in range(n_qs)]
    mags = []
    no_comps = []
    for q in master_qs:
        mag = q.get_magnitude()
        nc = q.get_num_components()
        print("Q:{}, difficulty={:.2f} across {} components".format(q.id, mag, nc))
        mags.append(mag)
        no_comps.append(nc)
    for q in master_qs:
        print("qid",q.id,q.betas)
    return master_qs

def create_students(n_students, theta_min, theta_max, nt):
    psi_list = [classes.Student(psix, theta_min,theta_max, nt=nt) for psix in range(n_students)]
    mags = []
    for psi in psi_list[0:30]:
        mag = psi.get_magnitude()
        nc = psi.get_num_components()
        print("{}, skill={:.2f} across {} comps".format(psi.name, mag, nc))
        mags.append(mag)

#     fig,ax = plt.subplots(1,2)
#     fig.set_size_inches(20,10)
#     ax[0].hist(mags)

#     if nt >1:
#         itemz = array([ s.thetas for s in psi_list ])
#         ax[1].scatter(itemz[:,0], itemz[:,1], alpha=0.2)
#         for i, txt in enumerate(itemz):
#             ax[1].annotate(i, (itemz[i,0], itemz[i,1]))
#         plt.show()
    return psi_list
    


def calculate_pass_probability(thetas, betas, gate_theta=False):
    p_pass = 1.0
    for th,b in zip(thetas,betas):
        if gate_theta and th==0:
            return 0
        p_pass_step=1.0 if (b==0) else (1.0 / (1.0 + exp(-(th-b))))
        p_pass *= p_pass_step # simple conjunctive model of success!
    try:
        pass
        #print("p_pass={}".format(p_pass))
    except OverflowError:
        p_pass = 0.0
    return p_pass

def attempt_q(student, q):
    p = calculate_pass_probability(student.thetas, q.betas)
    if (uniform(0,1) <= p):
#     if p >= 0.5:
        passed=1
    else:
        passed=0
    return p,passed

from collections import Counter
def generate_attempts(master_qs, psi_list):
    attempts =[]
    attempts_by_q = {}
    attempts_by_psi = {}
    attempt_n_map = Counter()
    user_budget = math.inf
    user_patience = 10 #math.inf
    pass_to_remove = False
    
#     master_betas  = numpy.array([q.betas() for q in master_qs])
#     master_alphas = numpy.array([s.thetas() for s in psi_list])
    
#     master_betas = numpy.repeat(master_betas, len(master_qs), axis=0 )
#     master_alphas = numpy.tile(master_alphas, (len(psi_list),1) )
    
#     diff_arr = master_alphas - master_betas
#     sigmoided = (lambda z: (1/1+numpy.exp(-z))) (diff_arr)
#     prs = numpy.prod(sigmoided, axis=1, keepdims=True)
    
    for run in range(1):
        print("----{}\n".format(run))
        psix = 0
        for psi in psi_list:
            if psix % 100 == 0:
                print(psix)
            psix+=1
            spend=0
#             qs = [ix for ix in range(len(master_qs))]
#             while qs:
#                 qix = choice(qs)
#                 q = master_qs[qix]
#                 passed=0
#                 if psi.name not in attempts_by_psi:
#                     attempts_by_psi[psi.name]=[]
#                 if q not in attempts_by_q:
#                     attempts_by_q[q]=[]
#                 pp,passed = attempt_q(psi, q)
#                 tup = (psi.id, q.id, passed, -math.log(pp))
#                 attempt_n_map[(q.id,psi.id)] += 1
#                 attempts.append(tup)
#                 attempts_by_psi[psi.name].append(tup)
#                 attempts_by_q[q].append(tup)
#                 qs.remove(qix)
            for qix,q in enumerate(master_qs):
                pp,passed = attempt_q(psi, q)
                tup = (psi.id, q.id, passed, -math.log(pp))
                attempts.append(tup)
    ct=0
    pct=0
    for a in attempts:
        ct+=1
        if a[2]:
            pct+=1
    print("passed {}/{}".format(pct,ct))    
    return attempts, attempts_by_q, attempts_by_psi, attempt_n_map