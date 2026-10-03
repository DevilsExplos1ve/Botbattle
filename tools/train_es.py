"""Evolution-strategies (OpenAI-ES style) policy search for bot.py's PARAMS.

Reward per game: win = 1 + 0.25*(1 - turns/1200) (faster wins better),
draw = 0.4, loss = 0. Opponents: the frozen current champion (self-play) and
the organizer expander. Antithetic gaussian perturbations in log/relative
space, rank-normalised fitness, Adam-free plain gradient ascent.
Writes the best parameters to tools/params_best.json after every generation.
"""
import json, os, random, sys, time, math
from multiprocessing import Pool

BOT = os.path.abspath("bot.py")
EXP = os.path.abspath("tools/expander.py")
KEYS_INT_SAFE = None

def init_worker():
    global arena
    sys.path.insert(0, os.path.abspath("tools"))
    import arena as _a
    arena = _a

def run_game(job):
    params, opp_params, opp_path, seed, swap = job
    me = (BOT, params)
    op = (opp_path, opp_params)
    a, b = (me, op) if not swap else (op, me)
    w, t, slow = arena.play(a, b, seed)
    idx = 0 if not swap else 1
    if w == idx:
        return 1.0 + 0.25 * (1 - t / 1200.0)
    if w == -1:
        return 0.4
    return 0.0

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pop_pairs = 4
    sigma = 0.15
    lr = 0.08
    sys.path.insert(0, os.path.abspath("tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("b0", BOT); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    base = dict(m.PARAMS)
    keys = sorted(base)
    theta = {k: float(v) for k, v in base.items()}
    champion = dict(theta)
    rng = random.Random(7)
    seed_ctr = 50000
    pool = Pool(4, initializer=init_worker)
    log = open("tools/es_log.txt", "a")
    for g in range(gens):
        t0 = time.time()
        eps_list = []
        for _ in range(pop_pairs):
            eps = {k: rng.gauss(0, 1) for k in keys}
            eps_list += [eps, {k: -v for k, v in eps.items()}]
        cands = [{k: theta[k] * math.exp(sigma * e[k]) if theta[k] > 0 else theta[k] + sigma * e[k] for k in keys} for e in eps_list]
        seeds = [seed_ctr + i for i in range(3)]; seed_ctr += 3
        jobs = []
        for ci, c in enumerate(cands):
            for s in seeds:
                for sw in (0, 1):
                    jobs.append((c, champion, BOT, s, sw))
            jobs.append((c, None, EXP, seeds[0], ci % 2))
        res = pool.map(run_game, jobs)
        per = len(jobs) // len(cands)
        fits = [sum(res[i * per:(i + 1) * per]) / per for i in range(len(cands))]
        order = sorted(range(len(cands)), key=lambda i: fits[i])
        ranks = [0.0] * len(cands)
        for r, i in enumerate(order):
            ranks[i] = r / (len(cands) - 1) - 0.5
        for k in keys:
            grad = sum(ranks[i] * eps_list[i][k] for i in range(len(cands))) / (len(cands) * sigma)
            if theta[k] > 0:
                theta[k] *= math.exp(lr * grad)
            else:
                theta[k] += lr * grad
        bi = order[-1]
        # champion update: best candidate becomes the new self-play opponent if it beat the champion clearly
        if fits[bi] > 0.75:
            champion = dict(cands[bi])
        msg = f"gen {g} mean_fit {sum(fits)/len(fits):.3f} best {fits[bi]:.3f} {time.time()-t0:.0f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        json.dump({"theta": theta, "champion": champion}, open("tools/params_best.json", "w"), indent=1)

if __name__ == "__main__":
    main()
