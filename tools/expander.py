"""Organizer reference expander strategy (competition/agents/expander_python), ported to act(dict)."""
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def act(obs):
    H, W = obs["height"], obs["width"]; T, O, A = obs["type"], obs["owner"], obs["army"]
    best, bm, fv = -1.0, None, None
    for r in range(H):
        for c in range(W):
            if O[r][c] != 1 or A[r][c] <= 1: continue
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or T[nr][nc] in (2, 5): continue
                m = [0, r, c, d, 0]
                if fv is None: fv = m
                if A[r][c] <= A[nr][nc] + 1: continue
                opp = O[nr][nc] == 2
                exp = opp or (O[nr][nc] == 0 and T[nr][nc] not in (0, 5))
                s = float(A[r][c]) * (10 if exp else 1) * (2 if opp else 1)
                if s > best: best, bm = s, m
    return bm or fv or [1, 0, 0, 0, 0]
