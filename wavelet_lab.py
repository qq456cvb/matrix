import math

def forward_no_normalization(v):
    D = {}
    while len(v) > 1:
        k = len(v)
        vnew = [(v[i] + v[i+1]) / 2 for i in range(0, k, 2)]
        w = [v[i] - v[i+1] for i in range(0, k, 2)]
        D.update(dict(zip([(k//2, i) for i in range(k//2)], w)))
        v = vnew
    D[(0, 0)] = v[0]
    return D


def normalize_coefficients(n, D):
    return dict((k, D[k] * math.sqrt(n / (4 * (k[0] if k[0] != 0 else 0.25)))) for k in D)


def forward(v):
    return normalize_coefficients(len(v), forward_no_normalization(v))


def suppress(D, threshold):
    return dict((k, 0 if abs(D[k]) < threshold else D[k]) for k in D)


def sparsity(D):
    return sum([1 if D[k] != 0 else 0 for k in D]) / len(D)


def unnormalize_coefficients(n, D):
    return dict((k, D[k] / math.sqrt(n / (4 * (k[0] if k[0] != 0 else 0.25)))) for k in D)


def backward_no_normalization(D):
    n = len(D)
    v = [D[(0, 0)]]
    while len(v) < n:
        k = 2 * len(v)
        v = [v[i // 2] - (i % 2 - 0.5) * D[(k // 2, i // 2)] for i in range(k)]
    return v


def backward(D):
    return backward_no_normalization(unnormalize_coefficients(len(D), D))


def dictlist_helper(dlist, k):
    return [d[k] for d in dlist]


def forward2d(vlist):
    D_list = [forward(v) for v in vlist]
    L_dict = dict((k, dictlist_helper(D_list, k)) for k in D_list[0])
    D_dict = dict((k, forward(L_dict[k])) for k in L_dict)
    return D_dict


def suppress2d(D_dict, threshold):
    return dict((kn, dict((km, 0 if abs(D_dict[kn][km]) < threshold else D_dict[kn][km]) for km in D_dict[kn])) for kn in D_dict)


def sparsity2d(D_dict):
    return sum([sum([1 if abs(D_dict[kn][km]) != 0 else 0 for km in D_dict[kn]]) for kn in D_dict]) / len(D_dict) / len(list(D_dict.values())[0])


def listdict2dict(L_dict, i):
    return dict((k, L_dict[k][i]) for k in L_dict)


def listdict2dictlist(listdict):
    return [listdict2dict(listdict, i) for i in range(len(list(listdict.values())[0]))]


def backward2d(dictdict):
    D_list = dict((k, backward(dictdict[k])) for k in dictdict)
    L_dict = listdict2dictlist(D_list)
    L_list = [backward(v) for v in L_dict]
    return L_list


def image_round(image):
    return [[min(255, abs(round(n))) for n in row] for row in image]

from image import file2image, color2gray, image2display
img = color2gray(file2image('Dali.png'))
# image2display(img)
wave_comp = forward2d(img)
wave_comp = suppress2d(wave_comp, 100)
print('sparsity is %f' % sparsity2d(wave_comp))
dump_img = image_round(backward2d(wave_comp))
image2display(dump_img)