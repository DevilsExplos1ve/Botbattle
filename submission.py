"""
Code Bot entry — "Castellan"
Registered ID: [PARTICIPANT_ID]   Bot name: Castellan

Strategy
--------
A parameterised policy over hand-built action features, with its weights tuned
by reinforcement learning (evolution strategies on win/draw/loss reward from
self-play and play against reference bots on the pinned competition engine;
see train_es.py in the development repo). The policy:

  1. Instant wins: capture / deathtouch (turn >= 800) the enemy general when an
     owned neighbour can execute onto it.
  2. Defence: capture enemy armies that come close to our general (from turn
     800 any enemy tile adjacent to our general is lethal, so it is removed
     with priority); pull armies home when a visible threat outguns the general.
  3. Economy: build castles (35 + proximity surcharge) on safe own cells,
     gathering army to a chosen site; expand onto neutral land.
  4. Scouting: remember terrain and every enemy cell seen; estimate the enemy
     general's location from the enemy territory seen so far.
  5. Pressure: when ahead on army, or as the deathtouch turn approaches, push
     the biggest army along a BFS path to the (estimated) enemy general.
  Every candidate move is scored by a linear function of features; the
  weights in PARAMS are the learned part.

Sources: organizer starter (examples/starter / expander agent) for the action
format; game semantics from the pinned engine generals-bots@13db8f6.
AI assistance: written with Claude Code (Anthropic), which drafted the code and
the evolution-strategies training harness; participant reviewed and tuned it.
Standard library only.
"""

from collections import deque

# ---- learned parameters (overwritten by the ES trainer) -------------------
PARAMS = {
    'attack_min': 44.9737,
    'attack_ratio': 0.5257,
    'castle_gen_dist': 3.8155,
    'castle_max': 6.7967,
    'castle_safe': 2.743,
    'castle_start': 32.9207,
    'def_margin': 2.3849,
    'def_radius': 4.344,
    'gen_hold_turn': 164.6497,
    'intercept_slack': 4.4644,
    'rush_turn': 660.2708,
    'w_enemy': 18.1933,
    'w_enemy_army': 0.1257,
    'w_far': 0.1197,
    'w_fog': 4.6324,
    'w_gather': 0.2566,
    'w_neutral': 16.8603,
    'w_small': 0.0341,
}

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10 ** 9


class Brain:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.blocked = [[False] * W for _ in range(H)]   # known mountain/structure
        self.seen = [[False] * W for _ in range(H)]
        self.enemy_seen = {}       # (r,c) -> last turn seen enemy-owned
        self.enemy_gen = None
        self.my_gen = None
        self.castles_built = 0
        self.site = None
        self.last_turn = -1

    # -------------------------------------------------------------- helpers
    def nbrs(self, r, c):
        H, W = self.H, self.W
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield d, nr, nc

    def bfs(self, sources, passable):
        H, W = self.H, self.W
        dist = [[INF] * W for _ in range(H)]
        q = deque()
        for (r, c) in sources:
            if dist[r][c] != 0:
                dist[r][c] = 0
                q.append((r, c))
        while q:
            r, c = q.popleft()
            nd = dist[r][c] + 1
            for _, nr, nc in self.nbrs(r, c):
                if dist[nr][nc] > nd and passable(nr, nc):
                    dist[nr][nc] = nd
                    q.append((nr, nc))
        return dist

    # --------------------------------------------------------------- policy
    def act(self, obs):
        P = PARAMS
        H, W = self.H, self.W
        turn = obs["turn"]
        T, O, A = obs["type"], obs["owner"], obs["army"]

        mine = []
        my_castles = []
        for r in range(H):
            Tr, Or = T[r], O[r]
            for c in range(W):
                t = Tr[c]
                if t != 0:
                    self.seen[r][c] = True
                if t == 2 or (t == 5 and Or[c] == 0):
                    self.blocked[r][c] = True
                elif t in (1, 3, 4):
                    self.blocked[r][c] = False
                o = Or[c]
                if o == 1:
                    mine.append((r, c))
                    if t == 4:
                        self.my_gen = (r, c)
                    elif t == 3:
                        my_castles.append((r, c))
                elif o == 2:
                    self.enemy_seen[(r, c)] = turn
                    if t == 4:
                        self.enemy_gen = (r, c)
                elif (r, c) in self.enemy_seen and t != 0:
                    del self.enemy_seen[(r, c)]
        if self.enemy_gen is not None:
            er, ec = self.enemy_gen
            if T[er][ec] != 0 and not (T[er][ec] == 4 and O[er][ec] == 2):
                self.enemy_gen = None
        if not mine:
            return [1, 0, 0, 0, 0]
        gen = self.my_gen or mine[0]
        blocked = self.blocked

        def passable(r, c):
            return not blocked[r][c]

        gdist = self.bfs([gen], passable)
        visible_enemy = []
        for (r, c) in self.enemy_seen:
            if O[r][c] == 2:
                visible_enemy.append((r, c))

        # ---- 1. instant win -------------------------------------------
        if self.enemy_gen is not None:
            er, ec = self.enemy_gen
            best = None
            for d, nr, nc in self.nbrs(er, ec):
                if O[nr][nc] == 1 and A[nr][nc] >= 2:
                    moved = A[nr][nc] - 1
                    if turn >= 800 or moved > A[er][ec]:
                        if best is None or A[nr][nc] > best[0]:
                            best = (A[nr][nc], nr, nc, d ^ 1)
            if best is not None:
                return [0, best[1], best[2], best[3], 0]

        # ---- 2. defence -------------------------------------------------
        gr, gc = gen
        threats = []
        for (r, c) in visible_enemy:
            dd = gdist[r][c]
            if dd <= P["def_radius"] and A[r][c] >= 2:
                threats.append((dd, -A[r][c], r, c))
        threats.sort()
        for dd, na, er, ec in threats:
            ea = -na
            # capture the threat from the best adjacent own cell
            best = None
            for d, nr, nc in self.nbrs(er, ec):
                if O[nr][nc] == 1 and A[nr][nc] - 1 > ea:
                    # prefer non-general sources unless adjacent to general
                    key = A[nr][nc] - (1000 if (nr, nc) == gen and dd > 1 else 0)
                    if best is None or key > best[0]:
                        best = (key, nr, nc, d ^ 1)
            if best is not None and (dd <= 2 or turn >= 780 or ea + P["def_margin"] >= A[gr][gc] or dd <= 3):
                return [0, best[1], best[2], best[3], 0]
            danger = (turn >= 760 and dd <= 3) or (ea + P["def_margin"] >= A[gr][gc] and dd <= P["def_radius"])
            if danger:
                # intercept with an army that can beat the threat
                tdist = self.bfs([(er, ec)], passable)
                inter = None
                for (r, c) in mine:
                    a = A[r][c]
                    if a - 1 > ea and (r, c) != gen and tdist[r][c] <= dd + P["intercept_slack"]:
                        k = (tdist[r][c], -a)
                        if inter is None or k < inter[0]:
                            inter = (k, r, c)
                if inter is not None:
                    _, r, c = inter
                    for d, nr, nc in self.nbrs(r, c):
                        if tdist[nr][nc] < tdist[r][c] and (O[nr][nc] == 1 or A[r][c] - 1 > A[nr][nc]):
                            return [0, r, c, d, 0]
                mv = self.gather_move(mine, A, O, gdist, gen, exclude_gen=True)
                if mv is not None:
                    return mv

        # ---- 3. castle economy --------------------------------------
        enemy_dist = None
        if visible_enemy or self.enemy_seen:
            enemy_dist = self.bfs(list(self.enemy_seen.keys()), passable)
        structs = [gen] + my_castles
        if turn >= P["castle_start"] and len(my_castles) < P["castle_max"]:
            # build where affordable
            for (r, c) in mine:
                if T[r][c] != 1:
                    continue
                cost = self.cost(r, c, structs)
                if A[r][c] >= cost and (enemy_dist is None or enemy_dist[r][c] >= P["castle_safe"] - 1):
                    self.castles_built += 1
                    self.site = None
                    return [2, r, c, 0, 0]
            # choose / keep a site
            if self.site is not None:
                sr, sc = self.site
                if O[sr][sc] != 1 or T[sr][sc] != 1:
                    self.site = None
            if self.site is None:
                best = None
                for (r, c) in mine:
                    if T[r][c] != 1:
                        continue
                    gd = gdist[r][c]
                    if gd >= INF:
                        continue
                    if enemy_dist is not None and enemy_dist[r][c] < P["castle_safe"]:
                        continue
                    cost = self.cost(r, c, structs)
                    score = -cost - 2.0 * abs(gd - P["castle_gen_dist"]) + 0.5 * A[r][c]
                    if best is None or score > best[0]:
                        best = (score, r, c)
                if best is not None:
                    self.site = (best[1], best[2])

        # ---- 4. target for pressure --------------------------------
        target = self.enemy_gen or self.guess_enemy_gen(gen, gdist, T)
        attack = False
        if target is not None:
            if turn >= P["rush_turn"]:
                attack = True
            elif obs["my_army"] > P["attack_ratio"] * obs["opp_army"] and obs["my_army"] > P["attack_min"]:
                attack = True
        if attack:
            tdist = self.bfs([target], passable)
            mv = self.push_move(mine, A, O, T, tdist, gen, turn)
            if mv is not None:
                return mv

        # ---- 5. scored greedy moves ---------------------------------
        site_dist = None
        if self.site is not None:
            site_dist = self.bfs([self.site], passable)
        best = None
        for (r, c) in mine:
            a = A[r][c]
            if a < 2:
                continue
            is_gen = (r, c) == gen
            split = 1 if (is_gen and turn >= P["gen_hold_turn"]) else 0
            moved = a // 2 if split else a - 1
            if moved < 1:
                continue
            for d, nr, nc in self.nbrs(r, c):
                if blocked[nr][nc]:
                    continue
                t = T[nr][nc]
                o = O[nr][nc]
                da = A[nr][nc]
                if o == 1:
                    if site_dist is None:
                        continue
                    if (r, c) == self.site:
                        continue
                    gain = site_dist[r][c] - site_dist[nr][nc]
                    if gain <= 0:
                        continue
                    s = P["w_gather"] * moved * gain
                    if is_gen and turn >= 700:
                        s *= 0.3
                elif o == 2:
                    if moved <= da:
                        continue
                    s = P["w_enemy"] + P["w_enemy_army"] * da
                    if t == 3:
                        s += 25.0
                else:
                    if moved <= da:
                        continue
                    if t == 0:
                        s = P["w_fog"]
                    else:
                        s = P["w_neutral"]
                    gd = gdist[nr][nc]
                    if gd < INF:
                        s += P["w_far"] * min(gd, 20)
                    s -= P["w_small"] * moved
                    if is_gen and turn >= 700:
                        s -= 5.0
                if best is None or s > best[0]:
                    best = (s, r, c, d, split)
        if best is not None:
            return [0, best[1], best[2], best[3], best[4]]
        return [1, 0, 0, 0, 0]

    # ------------------------------------------------------------------
    def cost(self, r, c, structs):
        cost = 35
        for (sr, sc) in structs:
            d = abs(sr - r) + abs(sc - c)
            if d < 7:
                cost += 14 - 2 * d
        return cost

    def gather_move(self, mine, A, O, dist, target, exclude_gen=False):
        best = None
        for (r, c) in mine:
            a = A[r][c]
            if a < 2 or (exclude_gen and (r, c) == target):
                continue
            for d, nr, nc in self.nbrs(r, c):
                if O[nr][nc] == 1 and dist[nr][nc] < dist[r][c]:
                    s = (a - 1) - 0.5 * dist[r][c]
                    if best is None or s > best[0]:
                        best = (s, r, c, d)
        if best is None:
            return None
        return [0, best[1], best[2], best[3], 0]

    def push_move(self, mine, A, O, T, tdist, gen, turn):
        # move the largest army that can make progress toward the target
        cands = sorted(((A[r][c], r, c) for (r, c) in mine if A[r][c] >= 2), reverse=True)
        for a, r, c in cands[:6]:
            if (r, c) == gen and turn < 800 and a < 40:
                continue
            best = None
            for d, nr, nc in self.nbrs(r, c):
                if tdist[nr][nc] >= tdist[r][c]:
                    continue
                if O[nr][nc] != 1 and a - 1 <= A[nr][nc]:
                    continue
                key = -tdist[nr][nc] * 10 + (1 if O[nr][nc] != 1 else 0)
                if best is None or key > best[0]:
                    best = (key, d)
            if best is not None:
                return [0, r, c, best[1], 0]
        return None

    def guess_enemy_gen(self, gen, gdist, T):
        H, W = self.H, self.W
        gr, gc = gen
        if self.enemy_seen:
            n = len(self.enemy_seen)
            tr = sum(p[0] for p in self.enemy_seen) / n
            tc = sum(p[1] for p in self.enemy_seen) / n
        else:
            tr, tc = H - 1 - gr, W - 1 - gc
        best = None
        for r in range(H):
            for c in range(W):
                gd = gdist[r][c]
                if gd >= INF:
                    continue
                if T[r][c] != 0:
                    continue  # only fog cells can hide the general
                s = ((r - tr) ** 2 + (c - tc) ** 2) ** 0.5 - 0.35 * min(gd, 30)
                if (r, c) in self.enemy_seen:
                    s -= 3.0
                if best is None or s < best[0]:
                    best = (s, r, c)
        if best is None:
            return None
        return (best[1], best[2])


_brain = None


def act(observation):
    global _brain
    try:
        H, W = int(observation["height"]), int(observation["width"])
        obs = dict(observation)
        for k in ("type", "owner", "army"):
            g = obs[k]
            if hasattr(g, "tolist"):
                g = g.tolist()
            if len(g) == H * W and not isinstance(g[0], (list, tuple)):
                g = [list(g[r * W:(r + 1) * W]) for r in range(H)]
            obs[k] = g
        observation = obs
        if _brain is None or _brain.H != H or _brain.W != W or observation["turn"] < _brain.last_turn:
            _brain = Brain(H, W)
        _brain.last_turn = observation["turn"]
        a = _brain.act(observation)
        return [int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4])]
    except Exception:
        return [1, 0, 0, 0, 0]
