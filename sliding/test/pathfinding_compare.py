import sys

def check_point(x, y, sx, sy):
    return 0 <= x and x < sx and 0 <= y and y <= sy

if __name__ == '__main__':
    input_data = sys.argv[1]
    output_data_1 = sys.argv[2]
    output_data_2 = sys.argv[3]

    with open(input_data, "r") as fd:
        lines = fd.readlines()

        size_x, size_y = tuple(map(int, lines[0].split(',')))

        weights = [[1 for _ in range(size_y)] for _ in range(size_x)]
        source_x, source_y = tuple(map(int, lines[1].split(',')))
        target_x, target_y = tuple(map(int, lines[2].split(',')))
        num_obstacles = int(lines[3])
        for i in range(4, 4 + num_obstacles):
            x, y = tuple(map(int, lines[i].split(',')))
            weights[x][y] = 0
        num_weigths = int(lines[4 + num_obstacles])
        for i in range(5 + num_obstacles, 5 + num_obstacles + num_weigths):
            x, y, w = tuple(map(int, lines[i].split(',')))
            weights[x][y] = w

    no_solution = False

    score_1 = 0
    with open(output_data_1, "r") as fd:
        lines = fd.readlines()

        if len(lines) == 1:
            no_solution = True
        else:
            x1, y1 = tuple(map(int, lines[1].split(',')))
            if not check_point(x1, y1, size_x, size_y):
                print('ORIGIN: Walked out of map: ({}, {}) not in ({}, {})'.format(x1, y1, size_x, size_y))
                sys.exit(1)
            if x1 != source_x or y1 != source_y:
                print('ORIGIN: Starting position is not a source')
                sys.exit(1)

            for i in range(2, len(lines)):
                x2, y2 = tuple(map(int, lines[i].split(',')))
                if not check_point(x2, y2, size_x, size_y):
                    print('ORIGIN: Walked out of map: ({}, {}) not in ({}, {})'.format(x2, y2, size_x, size_y))
                    sys.exit(1)
                if abs(x1 - x2) + abs(y1 - y2) != 1:
                    print('ORIGIN: Invalid step: from ({}, {}) to ({}, {})'.format(x1, y1, x2, y2))
                    sys.exit(1)
                if weights[x2][y2] == 0:
                    print('ORIGIN: Walked into obstacle ({}, {})'.format(x2, y2))
                    sys.exit(1)
                score_1 += weights[x2][y2]
                x1 = x2
                y1 = y2

            if x1 != target_x or y2 != target_y:
                print('ORIGIN: Ending position is not a target')
                sys.exit(1)


    score_2 = 0
    with open(output_data_2, "r") as fd:
        lines = fd.readlines()

        if len(lines) == 1:
            if not no_solution:
                sys.exit(1)
        else:
            x1, y1 = tuple(map(int, lines[1].split(',')))
            if not check_point(x1, y1, size_x, size_y):
                print('TEST: Walked out of map: ({}, {}) not in ({}, {})'.format(x1, y1, size_x, size_y))
                sys.exit(1)
            if x1 != source_x or y1 != source_y:
                sys.exit(1)

            for i in range(2, len(lines)):
                x2, y2 = tuple(map(int, lines[i].split(',')))
                if not check_point(x2, y2, size_x, size_y):
                    print('TEST: Walked out of map: ({}, {}) not in ({}, {})'.format(x2, y2, size_x, size_y))
                    sys.exit(1)
                if abs(x1 - x2) + abs(y1 - y2) != 1:
                    print('TEST: Invalid step: from ({}, {}) to ({}, {})'.format(x1, y1, x2, y2))
                    sys.exit(1)
                if weights[x2][y2] == 0:
                    print('TEST: Walked into obstacle ({}, {})'.format(x2, y2))
                    sys.exit(1)
                score_2 += weights[x2][y2]
                x1 = x2
                y1 = y2

            if x1 != target_x or y2 != target_y:
                print('TEST: Ending position is not a target')
                sys.exit(1)

    if score_1 != score_2:
        print('Scores not equal: origin = {}, test = {}'.format(score_1, score_2))
        sys.exit(1)

    sys.exit(0)

