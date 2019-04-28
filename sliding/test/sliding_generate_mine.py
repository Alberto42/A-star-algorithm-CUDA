import random
import sys
import copy

import numpy as np

def printArray(array):
    for i,a in enumerate(array):
        if a != 0:
            print(a,end='')
        else:
            print('_',end='')
        if i != n*n-1:
            print(',',end='')
        else:
            print()

n = int(sys.argv[1])
seed = int(sys.argv[2])
slides = int(sys.argv[3])
random.seed(seed)

array = np.array(range(n*n))
random.shuffle(array)
array = np.reshape(array,(n,n))

arrayStart = np.reshape(array,n*n).copy()

arrayTarget = arrayStart.copy()
while((arrayStart == arrayTarget).all()):
    for i in range(n):
        for j in range(n):
            if array[i][j] == 0:
                r,c = i,j

    moves = [[1,0],[-1,0],[0,1],[0,-1]]

    # print(array)
    for i in range(slides):
        while(True):
            (movR,movC) = random.choice(moves)
            newR,newC = r+movR, c+movC
            if (newR < 0 or newR >= n or newC < 0 or newC >= n):
                continue
            array[r][c],array[newR][newC] =array[newR][newC],array[r][c]
            r,c = newR,newC
            break
        # print(array)
    arrayTarget = np.reshape(array, n*n).copy()

printArray(arrayStart)
printArray(arrayTarget)


