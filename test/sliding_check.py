import sys
from math import sqrt

def f(s):
    if s == '_':
        return 0
    return int(s)

def compare(s1, s2):
    if len(s1) != len(s2):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return False
    return True

def move_possible(s1, s2):
    N = int(sqrt(len(s1)))
    for i in range(len(s1)):
        if s1[i] == 0:
            x1 = i // N
            y1 = i % N
    for i in range(len(s2)):
        if s2[i] == 0:
            x2 = i // N
            y2 = i % N

    if abs(x1 - x2) + abs(y1 - y2) != 1:
        return False

    cnt = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            cnt += 1

    if cnt != 2:
        return False

    if s1[x2 * N + y2] != s2[x1 * N + y1]:
        return False

    return True

if __name__ == '__main__':
    input_data = sys.argv[1]
    output_data = sys.argv[2]
    output_test_data = sys.argv[3]


    with open(input_data, "r") as fd:
        lines = fd.readlines()
        source = list(map(f, lines[0][:-1].split(',')))
        target = list(map(f, lines[1][:-1].split(',')))

    steps = []
    with open(output_data, "r") as fd:
        lines = fd.readlines()
        for i in range(1, len(lines)):
            steps.append(list(map(f, lines[i][:-1].split(','))))

    steps_test = []
    with open(output_test_data, "r") as fd:
        lines = fd.readlines()
        for i in range(1, len(lines)):
            steps_test.append(list(map(f, lines[i][:-1].split(','))))

    if len(steps_test) > 0:
        if not compare(source, steps_test[0]):
            print("TEST: First state is not source")
            sys.exit(1)
        if not compare(target, steps_test[len(steps_test) - 1]):
            print("TEST: Last state is not target")
            sys.exit(1)

        for i in range(1, len(steps_test)):
            if not move_possible(steps_test[i - 1], steps_test[i]):
                print("TEST: Move not possible from {} to {}".format(steps_test[i - 1], steps_test[i]))
                sys.exit(1)

    if len(steps) > 0:
        if not compare(source, steps[0]):
            print("ORIGIN: First state is not source")
            sys.exit(1)
        if not compare(target, steps[len(steps) - 1]):
            print("ORIGIN: Last state is not target")
            sys.exit(1)

        for i in range(1, len(steps)):
            if not move_possible(steps[i - 1], steps[i]):
                print("ORIGIN: Move not possible from {} to {}".format(steps[i - 1], steps[i]))
                sys.exit(1)

    if len(steps) != len(steps_test):
        print("Wrong solution")
        sys.exit(1)

    sys.exit(0)
