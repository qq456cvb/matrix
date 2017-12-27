# version code c2eb1c41017f+
# Please fill out this stencil and submit using the provided submission script.

import random
from GF2 import one
from vecutil import list2vec
from vec import dot
from independence import is_independent



## 1: (Task 7.7.1) Choosing a Secret Vector
def randGF2(): return random.randint(0,1)*one

def randGF2_6():
    return list2vec([randGF2(), randGF2(), randGF2(), randGF2(), randGF2(), randGF2()])

a0 = list2vec([one, one,   0, one,   0, one])
b0 = list2vec([one, one,   0,   0,   0, one])

def choose_secret_vector(s,t):
    u = randGF2_6()
    while dot(a0, u) != s or dot(b0, u) != t:
        u = randGF2_6()
    return u

print(choose_secret_vector(0, one))

## 2: (Task 7.7.2) Finding Secret Sharing Vectors
# Give each vector as a Vec instance
secret_a0 = list2vec([one, one,   0, one,   0, one])
secret_b0 = list2vec([one, one,   0,   0,   0, one])



for i in range(8):
    a1, b1, a2, b2 = [randGF2_6(), randGF2_6(), randGF2_6(), randGF2_6()]
    while (not is_independent([secret_a0, secret_b0, a1, b1, a2, b2])):
        a1, b1, a2, b2 = [randGF2_6(), randGF2_6(), randGF2_6(), randGF2_6()]

    a3, b3, a4, b4 = [randGF2_6(), randGF2_6(), randGF2_6(), randGF2_6()]
    while (not is_independent([secret_a0, secret_b0, a1, b1, a2, b2])) or \
            (not is_independent([secret_a0, secret_b0, a1, b1, a3, b3])) or \
            (not is_independent([secret_a0, secret_b0, a1, b1, a4, b4])) or \
            (not is_independent([secret_a0, secret_b0, a2, b2, a3, b3])) or \
            (not is_independent([secret_a0, secret_b0, a2, b2, a4, b4])) or \
            (not is_independent([secret_a0, secret_b0, a3, b3, a4, b4])) or \
            (not is_independent([a1, b1, a2, b2, a3, b3])) or \
            (not is_independent([a1, b1, a2, b2, a4, b4])) or \
            (not is_independent([a1, b1, a3, b3, a4, b4])) or \
            (not is_independent([a2, b2, a3, b3, a4, b4])):
        a3, b3, a4, b4 = [randGF2_6(), randGF2_6(), randGF2_6(), randGF2_6()]

secret_a1 = a1
secret_b1 = b1

secret_a2 = a2
secret_b2 = b2

secret_a3 = a3
secret_b3 = b3

secret_a4 = a4
secret_b4 = b4
