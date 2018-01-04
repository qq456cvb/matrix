from vec import Vec
from cancer_data import read_training_data
import matutil
from simplex import find_vertex, optimize

def main_constraint(i, a_i, d_i, features):
    labels = features | { i, 'gamma' }
    v = Vec(labels, {k: a_i[k] if d_i > 0 else -a_i[k] for k in features})
    if d_i > 0:
        v['gamma'] = -1
    else:
        v['gamma'] = 1
    v[i] = 1
    return v


def make_matrix(feature_vectors, diagnoises, features):
    main = dict((k, main_constraint(k, feature_vectors[k], diagnoises[k], features)) for k in feature_vectors.keys())
    minor = dict((-k, Vec(features | { k, 'gamma' }, {k: 1})) for k in feature_vectors.keys())
    main.update(minor)
    for v in main.values():
        v.D = features | set(feature_vectors.keys()) | { 'gamma' }
    #print(main[919555])
    #print(next(iter(main.values())))
    # print(main[-919555])
    return matutil.rowdict2mat(main)


def make_b(ids):
    return Vec(ids, {k: 1 if k > 0 else 0 for k in ids})


def make_c(features, ids):
    return Vec(features | ids | { 'gamma' }, {k: 1 for k in ids})

F, d = read_training_data('train.data')
# print(A.D[0])
features = {'texture(worst)', 'area(worst)'}
A = make_matrix(matutil.mat2rowdict(F), d, features)
b = make_b(A.D[0])
c = make_c(features, F.D[0])
R_square = F.D[0].copy()
n = len(A.D[1])
it = iter(F.D[0])
while len(R_square) < n:
    R_square.add(-next(it))

print(A.D[1])
find_vertex(A, b, R_square)
print(R_square)
print('vertex found')

x = optimize(A, b, c, R_square)
pred = Vec(F.D[0], {})
for k, r in matutil.mat2rowdict(F).items():
    s = 0
    for f in features:
        s += r[f] * x[f]
    if s > x['gamma']:
        pred[k] = 1
    else:
        pred[k] = -1
print(pred)
print(d)
# print(x['texture(worst)'])
# print(x['area(worst)'])
# print(x['gamma'])