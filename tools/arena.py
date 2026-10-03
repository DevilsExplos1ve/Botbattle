"""In-process match runner on the pinned competition engine (generals-bots@13db8f6).

Bots are single-file modules exposing act(observation); each game loads a fresh
copy of the module so globals reset, as in the official runner.
"""
import importlib.util, sys, time, os
import numpy as np

ENGINE = os.environ.get("GENERALS_ENGINE", "/tmp/generals-bots")
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, "competition"))
import jax
jax.config.update("jax_platform_name", "cpu")
from generals import GeneralsEnv
from generals.core import game
from matchup import make_board, make_transition

ENV = GeneralsEnv(mode="competition")
TRANSITION = jax.jit(make_transition(ENV))
GET_OBS = jax.jit(game.get_observation, static_argnums=1)

_counter = [0]
def load_bot(path, params=None):
    _counter[0] += 1
    spec = importlib.util.spec_from_file_location(f"bot_{_counter[0]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if params is not None and hasattr(mod, "PARAMS"):
        mod.PARAMS.update(params)
    return mod

def to_dict(obs, pid, H, W):
    armies = np.asarray(obs.armies)
    fog = np.asarray(obs.fog_cells); sf = np.asarray(obs.structures_in_fog)
    t = np.ones((H, W), dtype=np.int64)
    t[fog] = 0; t[sf] = 5
    t[np.asarray(obs.mountains, bool)] = 2
    t[np.asarray(obs.castles, bool)] = 3
    t[np.asarray(obs.generals, bool)] = 4
    o = np.zeros((H, W), dtype=np.int64)
    o[np.asarray(obs.owned_cells, bool)] = 1
    o[np.asarray(obs.opponent_cells, bool)] = 2
    return {"turn": int(obs.timestep), "height": H, "width": W, "player_id": pid,
            "my_land": int(obs.owned_land_count), "my_army": int(obs.owned_army_count),
            "opp_land": int(obs.opponent_land_count), "opp_army": int(obs.opponent_army_count),
            "type": t.tolist(), "owner": o.tolist(), "army": armies.astype(np.int64).tolist()}

def play(bot0, bot1, seed, max_ms=None, verbose=False):
    """bot0/bot1: (path, params). Returns (winner (-1 draw), turns, max_ms per bot)."""
    import jax.numpy as jnp
    mods = [load_bot(*bot0), load_bot(*bot1)]
    state = make_board(ENV, seed)
    H, W = (int(d) for d in state.armies.shape)
    slow = [0.0, 0.0]
    for turn in range(1300):
        acts = []
        for p in range(2):
            d = to_dict(GET_OBS(state, p), p, H, W)
            t0 = time.perf_counter()
            try:
                a = mods[p].act(d)
                ok = isinstance(a, (list, tuple)) and len(a) == 5 and all(type(x) is int for x in a)
            except Exception as e:
                if verbose: print("bot", p, "crashed", e)
                ok = False
            dt = (time.perf_counter() - t0) * 1000
            if turn > 0: slow[p] = max(slow[p], dt)
            if not ok:
                return 1 - p, turn, slow
            acts.append(a)
        state, info = TRANSITION(state, jnp.array(acts, dtype=jnp.int32))
        if bool(info.is_done) or int(state.time) >= 1200:
            return int(info.winner), int(state.time), slow
    return -1, int(state.time), slow

if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    res = []
    for s in range(n):
        for swap in (0, 1):
            pa, pb = (a, b) if not swap else (b, a)
            t0 = time.time()
            w, t, slow = play((pa, None), (pb, None), 1000 + s)
            me = (0 if not swap else 1)
            r = 1.0 if w == me else (0.5 if w == -1 else 0.0)
            res.append(r)
            print(f"seed {1000+s} swap {swap}: score {r} turns {t} slow_ms {slow[me]:.1f}/{slow[1-me]:.1f} wall {time.time()-t0:.1f}s", flush=True)
    print("mean score of", a, sum(res) / len(res))
