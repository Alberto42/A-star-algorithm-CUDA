import numpy as np
from math import sqrt

MIN_SIZE = 1000
MAX_SIZE = 1000
MAX_WEIGHTS = 5

def new_point(max_x, max_y, used):
    x = np.random.randint(max_x)
    y = np.random.randint(max_y)

    while (x, y) in used:
        x = np.random.randint(max_x)
        y = np.random.randint(max_y)

    used.add((x, y))

    return (x, y)

if __name__ == '__main__':
    sx = np.random.randint(MIN_SIZE, MAX_SIZE+1)
    sy = np.random.randint(MIN_SIZE, MAX_SIZE+1)
    N = sx * sy

    print('{},{}'.format(sx, sy))
    used = set()

    num_obstacles = np.random.randint(0, int(sqrt(N)))
    num_weights = np.random.randint(0, int(sqrt(N)))

    source_x, source_y = new_point(sx, sy, used)
    target_x, target_y = new_point(sx, sy, used)
    print('{},{}'.format(source_x, source_y))
    print('{},{}'.format(target_x, target_y))

    print(num_obstacles)
    for _ in range(num_obstacles):
        x, y = new_point(sx, sy, used)
        print('{},{}'.format(x, y))

    print(num_weights)
    for _ in range(num_weights):
        x, y = new_point(sx, sy, used)
        print('{},{},{}'.format(x, y, np.random.randint(1, MAX_WEIGHTS+1)))
