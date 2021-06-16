from random import choice, randint, random

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
