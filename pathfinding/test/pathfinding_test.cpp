#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <map>
#include <set>

#include <string>
#include <sstream>

using Pos = std::pair<int, int>;
using Qelem = std::pair<int, Pos>;
using World = std::vector<std::vector<int>>;
using PQueue = std::priority_queue<Qelem, std::vector<Qelem>, std::greater<Qelem>>;

int map_get(const std::map<Pos, int>& m, Pos key)
{
    if (m.find(key) == m.end())
        return 1;
    return m.at(key);
}

bool contains(const std::set<Pos>& s, Pos key)
{
    return s.find(key) != s.end();
}

int f(Pos s, Pos t)
{
    return std::abs(s.first - t.first) +
           std::abs(s.second - t.second);
}

bool push_or_update(std::map<Pos, int>& m, Pos key, int value)
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

std::map<Pos, Pos> astar(Pos grid, Pos start, Pos end, const World& weights)
{
    PQueue Q;
    Q.push(std::make_pair(0, start));
    std::set<Pos> H;
    std::map<Pos, int> g;
    g[start] = 0;

    std::map<Pos, Pos> path;

    while (!Q.empty())
    {
        Qelem elem = Q.top();
        Q.pop();
        Pos q = elem.second;
        int x = q.first, y = q.second;

        /* std::cout << x << " " << y << std::endl; */
        if (q == end)
            return path;

        if (!contains(H, q))
        {
            std::vector<Pos> next;
            if (x-1 >= 0 && weights[x-1][y] != 0)
                next.push_back(std::make_pair(x-1, y));
            if (x+1 < grid.first && weights[x+1][y] != 0)
                next.push_back(std::make_pair(x+1, y));
            if (y-1 >= 0 && weights[x][y-1])
                next.push_back(std::make_pair(x, y-1));
            if (y+1 < grid.second && weights[x][y+1])
                next.push_back(std::make_pair(x, y+1));

            for (const Pos& p : next)
            {
                if (!contains(H, p))
                {
                    int w = weights[p.first][p.second];
                    if (push_or_update(g, p, g[q] + w))
                        path[p] = q;
                    int value = f(p, end) + g[p];
                    Q.push(std::make_pair(value, p));
                }
            }

            H.insert(q);
        }

    }

    return std::map<Pos, Pos>();
}

int main(int argc, char** argv)
{
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);

    Pos grid, start, end;
    char kupa;

    std::ifstream file;
    file.open(input_path);

    file >> grid.first >> kupa >> grid.second;
    file >> start.first >> kupa >> start.second;
    file >> end.first >> kupa >> end.second;

    World world(grid.first);
    for (int x = 0; x < grid.first; ++x)
        world[x] = std::vector<int>(grid.second, 1);

    int num_obstacles;
    file >> num_obstacles;

    for (int i = 0; i < num_obstacles; ++i)
    {
        Pos o;
        file >> o.first >> kupa >> o.second;
        world[o.first][o.second] = 0;
    }

    int num_weights;
    file >> num_weights;
    for (int i = 0; i < num_weights; ++i)
    {
        Pos o;
        int v;
        file >> o.first >> kupa >> o.second >> kupa >> v;

        world[o.first][o.second] = v;
    }

    file.close();

    auto start_time = std::chrono::steady_clock::now();

    std::map<Pos, Pos> path_map = astar(grid, start, end, world);

    std::vector<Pos> path;
    if (!path_map.empty())
    {
        path.push_back(end);
        Pos current_pos = end;
        while (current_pos != start)
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
    for (Pos p : path)
        output_file << p.first << "," << p.second << "\n";

    output_file.close();

    return 0;
}
