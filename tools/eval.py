"""Head-to-head evaluation: candidate file vs opponent file(s), both slots, fresh seeds.
usage: python tools/eval.py cand.py opp.py [n_seeds] [opp_params_json]"""
import json, os, sys
from multiprocessing import Pool
def init():
    global arena
    sys.path.insert(0, os.path.abspath("tools")); import arena as a; arena = a
def game(j):
    cand, opp, oppp, seed, sw = j
    a, b = (cand, None), (opp, oppp)
    w, t, slow = arena.play(*((a, b) if not sw else (b, a)), seed)
    return (1.0 if w == sw else 0.5 if w == -1 else 0.0), max(slow)
if __name__ == "__main__":
    cand, opp = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    oppp = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    jobs = [(cand, opp, oppp, 90000 + s, sw) for s in range(n) for sw in (0, 1)]
    res = Pool(4, initializer=init).map(game, jobs)
    sc = [r for r, _ in res]
    print(f"{cand} vs {opp} {oppp}: {sum(sc)}/{len(sc)} = {sum(sc)/len(sc):.3f}  W/D/L {sc.count(1.0)}/{sc.count(0.5)}/{sc.count(0.0)}  max_ms {max(m for _, m in res):.1f}")
