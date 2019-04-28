import sys
import numpy as np

N = 3

def f(n):
    if n == 0:
        return "_"
    return str(n)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    k = N * N

    a1 = np.random.permutation(k)
    a2 = np.random.permutation(k)
    while((a1 == a2).all()):
        a1 = np.random.permutation(k)
        a2 = np.random.permutation(k)

    print(",".join(list(map(f, a1))))
    print(",".join(list(map(f, a2))))
