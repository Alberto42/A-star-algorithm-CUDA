#include <algorithm>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <queue>
#include <vector>
#include <set>
#include <map>

#ifndef N
#define N 3
#endif

struct State
{
    int grid[N * N];
    int empty_x, empty_y;

    bool operator==(const State& other) const
    {
        for (int i = 0; i < N * N; ++i)
            if (grid[i] != other.grid[i]) return false;
        return true;
    }

    bool operator!=(const State& other) const
    {
        return !operator==(other);
    }

    bool operator<(const State& other) const
    {
        for (int i = 0; i < N * N; ++i)
            if (grid[i] < other.grid[i])
                return true;
            else if (grid[i] > other.grid[i])
                return false;

        return false;
    }
};

std::ostream& operator<<(std::ostream& os, const State& state)
{
    for (int i = 0; i < N * N - 1; ++i)
        if (state.grid[i] == 0)
            os << "_,";
        else
            os << state.grid[i] << ",";

    if (state.grid[N * N - 1] == 0)
        os << "_";
    else
        os << state.grid[N * N - 1];

    return os;
}

using Qelem = std::pair<int, State>;
using PQueue = std::priority_queue<Qelem, std::vector<Qelem>, std::greater<Qelem>>;

bool contains(const std::set<State>& s, const State& key)
{
    return s.find(key) != s.end();
}

bool push_or_update(std::map<State, int>& m, const State& key, int value)
{
    if (m.find(key) == m.end())
    {
        m[key] = value;
        return true;
    }
    if (value < m[key])
    {
        m[key] = value;
        return true;
    }
    return false;
}

int f(const State& s1, const State& s2)
{
    int sum = 0;
    for (int x1 = 0; x1 < N; ++x1)
    {
        for (int y1 = 0; y1 < N; ++y1)
        {
            int v = s1.grid[x1 * N + y1];
            if (v == 0)
                continue;

            for (int pos = 0; pos < N * N; ++pos)
            {
                if (s2.grid[pos] == v)
                {
                    sum += std::abs(x1 - (pos / N)) + std::abs(y1 - (pos % N));
                    break;
                }
            }
        }
    }

    return sum;
}

std::map<State, State> astar(State source, State target)
{
    PQueue Q;
    std::set<State> H;
    std::map<State, int> g;
    std::map<State, State> path;

    Q.push(std::make_pair(0, source));
    g[source] = 0;

    while (!Q.empty())
    {
        Qelem elem = Q.top();
        Q.pop();

        State q = elem.second;

        if (q == target) return path;

        if (!contains(H, q))
        {
            if (q.empty_x - 1 >= 0)
            {
                int pos = q.empty_x * N + q.empty_y;
                State new_q = q;
                new_q.grid[pos] = new_q.grid[pos - N];
                new_q.grid[pos - N] = 0;
                new_q.empty_x--;

                if (push_or_update(g, new_q, g[q] + 1)) path[new_q] = q;
                int value = f(new_q, target) + g[new_q];
                Q.push(std::make_pair(value, new_q));
            }
            if (q.empty_x + 1 < N)
            {
                int pos = q.empty_x * N + q.empty_y;
                State new_q = q;
                new_q.grid[pos] = new_q.grid[pos + N];
                new_q.grid[pos + N] = 0;
                new_q.empty_x++;

                if (push_or_update(g, new_q, g[q] + 1)) path[new_q] = q;
                int value = f(new_q, target) + g[new_q];
                Q.push(std::make_pair(value, new_q));
            }
            if (q.empty_y - 1 >= 0)
            {
                int pos = q.empty_x * N + q.empty_y;
                State new_q = q;
                new_q.grid[pos] = new_q.grid[pos - 1];
                new_q.grid[pos - 1] = 0;
                new_q.empty_y--;

                if (push_or_update(g, new_q, g[q] + 1)) path[new_q] = q;
                int value = f(new_q, target) + g[new_q];
                Q.push(std::make_pair(value, new_q));
            }
            if (q.empty_y + 1 < N)
            {
                int pos = q.empty_x * N + q.empty_y;
                State new_q = q;
                new_q.grid[pos] = new_q.grid[pos + 1];
                new_q.grid[pos + 1] = 0;
                new_q.empty_y++;

                if (push_or_update(g, new_q, g[q] + 1)) path[new_q] = q;
                int value = f(new_q, target) + g[new_q];
                Q.push(std::make_pair(value, new_q));
            }

            H.insert(q);
        }
    }

    return std::map<State, State>();
}

int main(int argc, char** argv)
{
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);

    State source;
    State target;

    std::string delim(",");

    std::ifstream input_file;
    input_file.open(input_path);

    std::string line;

    getline(input_file, line);
    for (int i = 0; i < N * N - 1; ++i)
    {
        int pos = line.find(delim);
        std::string token = line.substr(0, pos);
        line.erase(0, pos + delim.length());

        if (token == "_")
        {
            source.grid[i] = 0;
            source.empty_x = i / N;
            source.empty_y = i % N;
        }
        else
            source.grid[i] = std::stoi(token);
    }
    if (line == "_")
    {
        source.grid[N * N - 1] = 0;
        source.empty_x = N - 1;
        source.empty_y = N - 1;
    }
    else
        source.grid[N * N - 1] = std::stoi(line);

    getline(input_file, line);
    for (int i = 0; i < N * N - 1; ++i)
    {
        int pos = line.find(delim);
        std::string token = line.substr(0, pos);
        line.erase(0, pos + delim.length());

        if (token == "_")
        {
            target.grid[i] = 0;
            target.empty_x = i / N;
            target.empty_y = i % N;
        }
        else
            target.grid[i] = std::stoi(token);
    }
    if (line == "_")
    {
        target.grid[N * N - 1] = 0;
        target.empty_x = N - 1;
        target.empty_y = N - 1;
    }
    else
        target.grid[N * N - 1] = std::stoi(line);

    input_file.close();

    auto start_time = std::chrono::steady_clock::now();

    std::map<State, State> path_map = astar(source, target);

    std::vector<State> path;
    if (!path_map.empty())
    {
        path.push_back(target);
        State current_pos = target;
        while (current_pos != source)
        {
            path.push_back(path_map[current_pos]);
            current_pos = path_map[current_pos];
        }

        std::reverse(path.begin(), path.end());
    }

    auto end_time = std::chrono::steady_clock::now();

    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    std::ofstream output_file;
    output_file.open(output_path);

    output_file << time_ms << "\n";
    for (const State& s : path)
        output_file << s << "\n";
    output_file.close();

    return 0;
}
