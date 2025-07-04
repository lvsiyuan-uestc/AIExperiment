import heapq


def input_state(n, name):
    print(f"请输入 {n * n} 个数字（用空格分隔），表示{name}状态，0 表示空位：")
    nums = list(map(int, input().split()))
    if len(nums) != n * n:
        raise ValueError("输入数字个数错误")
    return tuple(nums)


def is_solvable(state, n):
    inv_count = 0
    flat = [x for x in state if x != 0]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inv_count += 1
    if n % 2 == 1:
        return inv_count % 2 == 0
    else:
        row = n - (state.index(0) // n)
        return (inv_count + row) % 2 == 0


def manhattan_distance(state, goal, n):
    distance = 0
    for num in range(1, n * n):
        idx_state = state.index(num)
        idx_goal = goal.index(num)
        x1, y1 = divmod(idx_state, n)
        x2, y2 = divmod(idx_goal, n)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance


def misplaced_tiles(state, goal, n):
    return sum(1 for i in range(n * n) if state[i] != goal[i] and state[i] != 0)


class PuzzleNode:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def get_neighbors(state, n):
    neighbors = []
    index = state.index(0)
    x, y = divmod(index, n)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            new_index = nx * n + ny
            new_state = list(state)
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            neighbors.append(tuple(new_state))
    return neighbors


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]


def a_star(start, goal, n, heuristic):
    open_list = []
    closed_set = set()
    h = heuristic(start, goal, n)
    start_node = PuzzleNode(start, None, 0, h)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.state == goal:
            return reconstruct_path(current_node)
        closed_set.add(current_node.state)
        for neighbor in get_neighbors(current_node.state, n):
            if neighbor in closed_set:
                continue
            g = current_node.g + 1
            h = heuristic(neighbor, goal, n)
            neighbor_node = PuzzleNode(neighbor, current_node, g, h)
            heapq.heappush(open_list, neighbor_node)
    return None


def print_state(state, n):
    for i in range(0, n * n, n):
        row = state[i:i + n]
        print(' '.join(str(num) if num != 0 else ' ' for num in row))
    print('---------')


def main():
    n = int(input("请输入拼图边长（例如 3 表示 3x3 拼图）："))
    start = input_state(n, "初始")
    goal = input_state(n, "目标")

    if is_solvable(start, n) != is_solvable(goal, n):
        print("该问题无解。")
        return

    print("请选择启发式函数：\n1. 曼哈顿距离\n2. 错位数")
    h_choice = input("输入数字 1 或 2：")
    heuristic = manhattan_distance if h_choice == "1" else misplaced_tiles

    solution = a_star(start, goal, n, heuristic)
    if solution:
        print(f"总步数: {len(solution) - 1}")
        for i, step in enumerate(solution):
            print(f"Step {i}:")
            print_state(step, n)
    else:
        print("未找到解决方案。")
    input("程序运行完成，按回车键退出...")


if __name__ == "__main__":
    main()
