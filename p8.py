from collections import deque

g = [
    [" ", ".", ".", ".", "."],
    [".", "X", "X", "G", "."],
    [".", ".", ".", " ", "."],
    ["S", ".", ".", ".", "."]
]

start = (3, 0)
goal = (1, 3)

R = len(g)
C = len(g[0])

dirs = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

q = deque([start])
parent = {start: None}

while q:
    r, c = q.popleft()
    if (r, c) == goal:
        break
    for move, (dr, dc) in dirs.items():
        nr = r + dr
        nc = c + dc
        if (0 <= nr < R and
            0 <= nc < C and
            g[nr][nc] != "X" and
            (nr, nc) not in parent):
            parent[(nr, nc)] = ((r, c), move)
            q.append((nr, nc))

path = []
cur = goal
while parent[cur]:
    cur, move = parent[cur]
    path.append(move)
path.reverse()
print("Path:", path)