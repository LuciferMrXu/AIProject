'''
搜素+路径排序：路径规划，自动决策
'''
# (0, 0), 90, 40
# (60, -), (-, 60)
from icecream import ic


def successors(x, y, X, Y):
    return {
        (0, y): '倒空x',
        (x, 0): '倒空y',
        (x + y - Y, Y) if x + y >= Y else (0, x + y): 'x倒入y中',
        (X, x + y - X) if y + x >= X else (x + y, 0): 'y倒入x中',
        (X, y): '装满x',
        (x, Y): '装满y'
    }


def search_solution(capacity1, capacity2, goal, start=(0, 0)):
    paths = [ [('init', start)] ]

    explored = set()

    while paths:
        path = paths.pop(0) # 宽度优先搜索BFS
        # path = paths.pop(-1) # 深度优先搜索DFS
        frontier = path[-1]
        (x, y) = frontier[-1]

        for state, action in successors(x, y, capacity1, capacity2).items():
            # ic(frontier, state, action)
            if state in explored: continue

            new_path = path + [ (action, state) ]

            if goal in state:
                return new_path
            else:
                paths.append(new_path)

            explored.add(state)

    return None


if __name__ == '__main__':
    path = search_solution(90, 40, 70, (0, 0))

    for p in path:
        print('--=>')
        print(p)

