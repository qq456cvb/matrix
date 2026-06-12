# Coding the Matrix — Lab Solutions

My completed labs for **_Coding the Matrix: Linear Algebra through Computer Science Applications_** by Philip Klein (Brown CS053, also offered on Coursera). Every lab in the book is finished.

The course's signature approach is building linear algebra from scratch in Python — the `Vec`/`Mat` classes here (sparse dictionary-backed vectors and matrices) are implemented as part of the exercises and then reused by every application lab.

## Labs

| Lab | Topic |
|-----|-------|
| `The_Vector_Space_problems.py`, `The_Matrix_problems.py` | core vector/matrix exercises |
| `GF2.py`, `ecc_lab.py` | error-correcting (Hamming) codes over GF(2) |
| `secret_sharing_lab.py` | threshold secret sharing over GF(2) |
| `factoring_lab.py` | integer factorization (quadratic-sieve style) |
| `Orthogonalization_problems.py`, `independence.py`, `echelon.py` | Gram–Schmidt, rank, echelon form |
| `eigenfaces_lab.py` (`faces/`, `unclassified/`) | eigenfaces face classification |
| `pagerank_lab.py` | PageRank power iteration |
| `machine_learning_lab.py` (`train.data`, `validate.data`) | linear classifier on the breast-cancer dataset |
| `perspective_lab.py` (`board.png`, `cit.png`) | perspective rectification via change of basis |
| `geometry_lab.py` | 2D transformations on images |
| `wavelet_lab.py` (`Dali.png`, `flag.png`) | Haar wavelet image compression |
| `linearprogramming_lab.py`, `simplex.py` | linear programming / simplex |
| `digits_lab.py` | handwritten digit experiments |
| `svd.py`, `solver.py`, `png.py`, `image*.py`, `*util.py` | support modules provided with the book |

## Usage

Python 3, standard library only (image I/O goes through the bundled pure-Python `png.py`). Run any lab directly, e.g. `python eigenfaces_lab.py`; the needed data files and images are committed alongside.

## Note

If you're currently taking CS053 or the Coursera course, solve the labs yourself before reading these.
