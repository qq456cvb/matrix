# version code 53ead35ddb8a+
# Please fill out this stencil and submit using the provided submission script.

from vec import Vec
from mat import Mat, equal
from math import sqrt
import pagerank
import pagerank_test



## 1: (Task 12.12.1) Find Number of Links
def find_num_links(L):
    '''
    Input:
        - L: a square matrix representing link structure
    Output:
        - A vector mapping each column label of L to
          the number of non-zero entries in the corresponding
          column of L
    Example:
        >>> from matutil import listlist2mat
        >>> find_num_links(listlist2mat([[1,1,1],[1,1,0],[1,0,0]]))
        Vec({0, 1, 2},{0: 3, 1: 2, 2: 1})
    '''
    return Vec(L.D[1], dict((c, sum([1 if L[r, c] == 1 else 0 for r in L.D[0]])) for c in L.D[1]))


## 2: (Task 12.12.2) Make Markov
def make_Markov(L):
    '''
    Input:
        - L: a square matrix representing link structure
    Output:
        - None: changes L so that it plays the role of A_1
    Example:
        >>> from matutil import listlist2mat
        >>> M = listlist2mat([[1,1,1],[1,0,0],[1,0,1]])
        >>> make_Markov(M)
        >>> M
        Mat(({0, 1, 2}, {0, 1, 2}), {(0, 1): 1.0, (2, 0): 0.3333333333333333, (0, 0): 0.3333333333333333, (2, 2): 0.5, (1, 0): 0.3333333333333333, (0, 2): 0.5})
    '''
    col_cnt = find_num_links(L)
    for c in L.D[1]:
        for r in L.D[0]:
            L[r, c] /= col_cnt[c]


## 3: (Task 12.12.3) Power Method
def power_method(A1, i):
    '''
    Input:
        - A1: a matrix
        - i: number of iterations to perform
    Output:
        - An approximation to the stationary distribution
    Example:
        >>> from matutil import listlist2mat
        >>> power_method(listlist2mat([[0.6,0.5],[0.4,0.5]]), 10)
        Vec({0, 1},{0: 0.5464480874307794, 1: 0.45355191256922034})
    '''
    v = Vec(A1.D[1], dict((k, 1) for k in A1.D[1]))
    nrows = len(A1.D[0])
    for _ in range(i):
        a2 = sum(v.f.values()) / nrows
        v = 0.85 * A1 * v + 0.15 * Vec(v.D, dict((k, a2) for k in v.D))
    return v

links = pagerank.read_data()

## 4: (Task 12.12.4) Jordan
# number_of_docs_with_jordan = len(pagerank.find_word('jordan'))



## 5: (Task 12.12.5) Wikigoogle
def wikigoogle(w, k, p):
    '''
    Input:
        - w: a word
        - k: number of results
        - p: pagerank eigenvector
    Output:
        - the list of the names of the kth heighest-pagerank Wikipedia
          articles containing the word w
    '''
    related = pagerank.find_word(w)
    related.sort(key=lambda x : p[x], reverse=True)
    return related[:k]



## 6: (Task 12.12.6) Using Power Method
p = power_method(links, 10)
results_for_jordan = wikigoogle('jordan', 5, p) # give 5 of them as a list
print(results_for_jordan)
# results_for_obama  = wikigoogle('obama', 5, p)
# results_for_tiger  = wikigoogle('tiger', 5, p)
# results_for_matrix = wikigoogle('matrix', 5, p)




## 7: (Task 12.12.7) Power Method Biased
def power_method_biased(A1, i, r):
    '''
    Input:
        - A1: a matrix, as in power_method
        - i: number of iterations
        - r: bias label
    Output:
        - Approximate eigenvector of .55A_1 + 0.15A_2 + 0.3A_r
    '''
    v = Vec(A1.D[1], dict((k, 1) for k in A1.D[1]))
    for _ in range(i):
        vsum = sum(v.f.values())
        a2 = vsum / len(A1.D[0])
        v = 0.55 * A1 * v + 0.15 * Vec(v.D, dict((k, a2) for k in v.D))
        v[r] += 0.3 * vsum
    return v
    

# p_sport = power_method_biased(links, 10, 'sport')
# sporty_results_for_jordan = wikigoogle('jordan', 5, p_sport)
# sporty_results_for_obama  = wikigoogle('obama', 5, p_sport)
# sporty_results_for_tiger  = wikigoogle('tiger', 5, p_sport)
# sporty_results_for_matrix = wikigoogle('matrix', 5, p_sport)

