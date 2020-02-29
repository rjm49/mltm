from random import uniform
from math import sqrt
class Question():
    def __init__(self, qix, min_diff, max_diff, nt, min_active, max_active):
        self.id = qix
        print("Qinit",qix)

        n_c = nt #randint(min_active, max_active)
        choices = range(nt) # numpy.random.choice(range(nt), size=n_c, replace=False)

        print("Q choices are {}".format(choices))

        not_present= 0 #min_diff
        self.betas = [ not_present for _ in range(nt) ]        

        for c in choices:
#             self.betas[c] = min_diff
#             self.betas[c] = random.uniform(min_diff, max_diff)
            self.betas[c]= uniform(min_diff, max_diff) #(1,11)
    def get_magnitude(self):
        comps = [c for c in self.betas if c>0]
        mag = sqrt(sum([ pow(b, 2) for b in comps ]))
        return mag
    def get_num_components(self):
        return len([c for c in self.betas if c>0])

from utils import generate_student_name
class Student():
    def __init__(self, ix, min_a, max_a, nt):
        self.id = ix
        self.name = generate_student_name()
        n_c = nt
        choices = range(nt) #numpy.random.choice(range(nt), size=n_c, replace=False)
        not_present= 0 #min_a
        self.thetas = [ not_present for _ in range(nt) ]        

        for c in choices:
            self.thetas[c] = uniform(min_a, max_a) #(7,22)
    def get_magnitude(self):
        comps = [c for c in self.thetas if c>0]
        mag = sqrt(sum([ pow(b, 2) for b in comps ]))
        return mag
    def get_num_components(self):
        return len([c for c in self.thetas if c>0])
        
