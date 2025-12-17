import random
from collections import deque
from typing import Callable, Dict, Iterable, Set, Tuple, Any

def polarity_aware_diffusion(H,
                             S: Iterable[Any],
                             polarity,
                             theta: float = 0.5,
                             rng: random.Random = None) -> Set[Any]:
    """
    Algorithm 3: Polarity-aware Influence Diffusion

    Parameters
    ----------
    H : HyperNetX Hypergraph (or any object where H.edges[e] -> set of users and H.nodes lists users)
        - Access a hyperedge's member set via H.edges[e]
    S : iterable
        Initial seed set (iterable of user ids)
    polarity : callable or mapping
        - If callable: polarity(u, e) -> some comparable polarity value (e.g. -1/0/1, 'pos'/'neg', True/False)
        - If mapping/dict: either
            * mapping[(u, e)] -> value, or
            * mapping[u][e] -> value
    theta : float in [0,1]
        Threshold for random chance to infect when polarities differ.
    rng : random.Random (optional)
        Random generator (useful for reproducibility). If None, uses random module.

    Returns
    -------
    A : set
        Set of activated users after the diffusion finishes.
    """

    # helper to get polarity value robustly
    def _pol(u, e):
        if callable(polarity):
            return polarity(u, e)
        # mapping-like handling:
        # try (u,e) tuple first
        try:
            return polarity[(u, e)]
        except Exception:
            pass
        # try nested: polarity[u][e]
        try:
            return polarity[u][e]
        except Exception:
            pass
        # try polarity.get(u, {}).get(e)
        try:
            return polarity.get(u, {}).get(e)
        except Exception:
            return None

    if rng is None:
        rand = random.random
    else:
        rand = rng.random

    # activated set A and queue Q
    A: Set[Any] = set(S)
    Q = deque(S)

    # main loop
    while Q:
        u = Q.popleft()

        # iterate over hyperedges incident on u.
        # HyperNetX HypergraphView does not provide .values(), but you can iterate edge names and index:
        for e in H.edges:            # e is the hyperedge key/name
            members = H.edges[e]     # set-like of users in this hyperedge
            if u not in members:
                continue

            # for each user v in this hyperedge that is not already activated
            for v in members:
                if v in A:
                    continue

                pol_u = _pol(u, e)
                pol_v = _pol(v, e)

                # if polarities equal (including both None) -> immediate activation
                if pol_u == pol_v:
                    A.add(v)
                    Q.append(v)
                else:
                    # polarity differs (or one missing) -> probabilistic activation
                    if rand() > theta:
                        A.add(v)
                        Q.append(v)

    return A


def relevance_based_seed_selection(H, T_prime, r, k):
    """
    Implements Algorithm 2: Relevance-Based Seed Selection

    Parameters:
    -----------
    H : hypernetx Hypergraph
        H.nodes  = users U
        H.edges  = topics (hyperedges)

    T_prime : list/set of selected topics (subset of H.edges)

    r : dict or map
        Relevance vector where r[t] gives relevance score of topic t

    k : int
        Number of seeds to select

    Returns:
    --------
    S : list
        Top-k seed users
    """

    # Users U
    U = list(H.nodes)

    # Dictionary to store relevance score of each user
    R = {u: 0.0 for u in U}

    # For each user u in U
    for u in U:

        # For each topic t in T'
        for t in T_prime:

            # Check if this topic exists in the hypergraph
            if t not in H.edges:
                continue

            # Hyperedge node set
            topic_users = H.edges[t]

            # If t is incident on u  (u belongs to hyperedge t)
            if u in topic_users:
                # w(u,t) = hyperparameter or simply |edge|
                w_ut = len(topic_users)

                # r_t = relevance score of topic t
                r_t = r.get(t, 0)

                # Update relevance score
                R[u] += w_ut * r_t

    # Sort users by descending relevance score |R[u]|
    sorted_users = sorted(U, key=lambda u: abs(R[u]), reverse=True)

    # Select top-k
    return sorted_users[:k]


def opinion_based_seed_selection(H, k):
    # Accept either a HyperNetX Hypergraph object or a tuple (V, E)
    if hasattr(H, "nodes") and hasattr(H, "edges") and not isinstance(H, tuple):
        V = list(H.nodes)
        E = H.edges
    else:
        V, E = H

    # initialize hyperdegree
    hyperdegree = {u: 0 for u in V}

    # iterate over edge names and fetch the node set for each edge
    for edge_name in E:
        edge_users = E[edge_name]   # works for HypergraphView
        for u in edge_users:
            # be safe if some user appears in edges but not in V
            hyperdegree[u] = hyperdegree.get(u, 0) + 1

    # sort by hyperdegree and return top-k
    sorted_users = sorted(hyperdegree.keys(), key=lambda u: hyperdegree[u], reverse=True)
    return sorted_users[:k]


# Linear Threshold Model Hypergraphs

import numpy as np


def LT_hypergraph(H, seed_set, th_low=0.0, th_high=0.1, mc=10):
    """
    Linear Threshold model on a Hypergraph.

    H : HyperNetX Hypergraph object
        H.nodes  -> users
        H.edges[e] -> set of users in hyperedge e

    seed_set : list or set
        Initial infected/activated users

    th_low, th_high : floats
        Threshold range (random per iteration)
        Threshold = Uniform(th_low, th_high)

    mc : int
        Number of Monte Carlo simulations

    Returns:
        mean spread over MC runs
    """

    spread = []
    nodes = list(H.nodes)

    for sim in range(mc):
        curr_active = set(seed_set)
        new_active = list(seed_set)

        # random threshold for this simulation
        np.random.seed(sim)
        threshold = np.random.uniform(th_low, th_high)
        thr_value = threshold * len(nodes)

        while new_active:
            newly_added = []

            # try activating all nodes not yet active
            for u in nodes:
                if u in curr_active:
                    continue

                # count active neighbors across all incident hyperedges
                active_neighbors = 0

                for e in H.edges:
                    if u not in H.edges[e]:
                        continue

                    # users in the same hyperedge
                    for v in H.edges[e]:
                        if v in curr_active and v != u:
                            active_neighbors += 1

                if active_neighbors > thr_value:
                    newly_added.append(u)

            new_active = newly_added
            curr_active.update(newly_added)

        spread.append(len(curr_active))

    return np.mean(spread)


#Independent Cascade Model Hypergraphs

import numpy as np

def IC_hypergraph(H, S, p=0.01, mc=10):
    """
    Independent Cascade (IC) model on a hypergraph.

    Parameters
    ----------
    H  : hypernetx.Hypergraph
         Nodes = users, hyperedges = opinion categories.
    S  : iterable
         Initial active seed nodes (user ids).
    p  : float
         Activation probability on each user–user influence attempt.
    mc : int
         Number of Monte Carlo simulations.

    Returns
    -------
    float
        Average spread (number of activated nodes).
    """

    # ---------- Build 2-section neighbors: users sharing a hyperedge ----------
    neighbors = {u: set() for u in H.nodes}

    for e in H.edges:          # e is hyperedge name
        users_in_e = set(H.edges[e])
        for u in users_in_e:
            neighbors[u].update(users_in_e - {u})

    spreads = []

    for i in range(mc):
        # For reproducibility per simulation
        rng = np.random.RandomState(i)

        curr_active = set(S)        # all active so far
        new_active = set(S)         # frontier in this step

        while new_active:
            next_active = set()

            for u in new_active:
                for v in neighbors[u]:
                    if v in curr_active:
                        continue

                    # each active u gets ONE chance to activate each neighbor v
                    if rng.uniform(0, 1) <= p:
                        next_active.add(v)

            # new frontier: those just activated and not previously active
            new_active = next_active - curr_active
            curr_active |= new_active

        spreads.append(len(curr_active))

    return float(np.mean(spreads))


import time

import time

# def greedyIC_hypergraph(H, k, p=0.1, mc=100):
#     """
#     Greedy hill-climbing seed selection under IC diffusion on a hypergraph.
#
#     Assumes you already have:
#         IC(H, S, p=0.1, mc=100) -> expected spread (float)
#
#     Parameters
#     ----------
#     H  : hypernetx.Hypergraph
#     k  : int          number of seeds to select
#     p  : float        IC activation probability
#     mc : int          Monte Carlo simulations (keep small at first!)
#
#     Returns
#     -------
#     S         : list  selected seeds
#     spread    : list  spread after each added seed
#     timeLapse : list  elapsed time (seconds) after each iteration
#     """
#     S = []
#     spread = []
#     timeLapse = []
#     startTime = time.time()
#
#     # spread of empty seed set
#     currentSpread = IC_hypergraph(H, S, p, mc)
#
#     for i in range(k):
#         bestNode = None
#         bestSpread = currentSpread
#
#         candidates = set(H.nodes) - set(S)
#
#         # if no candidates left, break early
#         if not candidates:
#             break
#
#         for j in candidates:
#             newSpread = IC_hypergraph(H, S + [j], p, mc)
#             if newSpread > bestSpread:
#                 bestSpread = newSpread
#                 bestNode = j
#
#         if bestNode is None:
#             # no candidate improved the spread; stop
#             break
#
#         S.append(bestNode)
#         currentSpread = bestSpread
#
#         spread.append(currentSpread)
#         timeLapse.append(time.time() - startTime)
#
#         # progress indicator so you see it's not stuck
#         print(f"[greedyIC] chosen {i+1}-th seed = {bestNode}, "
#               f"spread ≈ {currentSpread:.2f}, "
#               f"elapsed = {timeLapse[-1]:.1f}s")
#
#     return S, spread, timeLapse


import time
import numpy as np

def greedyIC_hypergraph(H, k, p=0.1, mc=1000):
    """
    Greedy hill-climbing seed selection under IC diffusion on a hypergraph.

    Parameters
    ----------
    H  : hypernetx.Hypergraph
         Nodes = users; hyperedges = opinion categories.
    k  : int
         Number of seeds to pick.
    p  : float
         Activation probability in IC model.
    mc : int
         Monte Carlo simulations.

    Returns
    -------
    (S, spread, timeLapse)
    S         : list of selected seeds
    spread    : list of best spread after each iteration
    timeLapse : elapsed time after each iteration
    """

    # ---- Precompute 2-section neighbors once (same as in IC_hypergraph) ----
    neighbors = {u: set() for u in H.nodes}
    print("Computing initial marginal gains...")
    for e in H.edges:
        users_in_e = set(H.edges[e])
        for u in users_in_e:
            neighbors[u].update(users_in_e - {u})

    # ---- IC function using precomputed neighbors ----
    def IC_fast(seed_set):
        spreads = []
        for sim in range(mc):
            rng = np.random.RandomState(sim)

            curr_active = set(seed_set)
            new_active = set(seed_set)

            while new_active:
                next_active = set()

                for u in new_active:
                    for v in neighbors[u]:
                        if v in curr_active:
                            continue
                        if rng.uniform(0, 1) <= p:
                            next_active.add(v)

                new_active = next_active - curr_active
                curr_active |= new_active

            spreads.append(len(curr_active))

        return np.mean(spreads)

    # ---- Greedy selection ----
    S = []
    spread = []
    timeLapse = []
    startTime = time.time()

    for i in range(k):
        bestSpread = 0
        bestNode = None

        currentSpread = IC_fast(S)
        print(currentSpread)
        for candidate in set(H.nodes) - set(S):

            newSpread = IC_fast(S + [candidate])
            marginalSpread = newSpread - currentSpread
            print(marginalSpread)
            if marginalSpread > bestSpread:
                bestSpread = marginalSpread
                bestNode = candidate

        if bestNode is not None:
            S.append(bestNode)

        spread.append(bestSpread)
        timeLapse.append(time.time() - startTime)

    return S, spread, timeLapse



##CELF Independent Cascade

import time
import numpy as np

def CELF_IC_hypergraph(H, k, p=0.01, mc=10, spread_func=None):
    """
    CELF optimization for Independent Cascade on a hypergraph.

    Parameters
    ----------
    H : hypernetx.Hypergraph
        Your opinionated hypergraph (nodes = users, edges = opinion hyperedges).
    k : int
        Number of seeds to select.
    p : float
        Activation probability in IC.
    mc : int
        Number of Monte Carlo simulations for IC.
    spread_func : callable or None
        Function that returns expected spread for a given seed set.
        If None, defaults to IC_hypergraph(H, S, p, mc).

    Returns
    -------
    S : list
        Final seed set of size k.
    timeLapse : list of float
        timeLapse[i] = elapsed time (seconds) after selecting (i+1)-th seed.
    final_mean_spread : float
        Expected spread (mean over mc runs) for final seed set S.
    """

    # Use IC on hypergraph as default spread function
    if spread_func is None:
        def spread_func(S):
            return IC_hypergraph(H, S, p=p, mc=mc)

    startTime = time.time()

    # Current seed set, spread and time history
    S = []
    timeLapse = []
    current_spread = 0.0

    # CELF priority queue: list of [node, marginal_gain, last_updated_seed_set_size]
    Q = []

    # ---------- 1. Initial marginal gains (with S = ∅) ----------
    print("Computing initial marginal gains...")
    for u in H.nodes:
        # gain = spread({u}) - spread(∅) = spread({u})
        mg = spread_func([u])
        Q.append([u, mg, 0])
        print(Q)

    # sort by marginal gain descending
    Q.sort(key=lambda x: x[1], reverse=True)

    # ---------- 2. Main CELF loop ----------
    for i in range(k):
        print(S)
        while True:
            v, mg, last = Q[0]

            # if this node's marginal gain was computed for the current S, accept it
            if last == len(S):
                break

            # otherwise, recompute its marginal gain w.r.t. current S
            new_spread = spread_func(S + [v])
            new_mg = new_spread - current_spread

            # update top element and re-sort
            Q[0] = [v, new_mg, len(S)]
            Q.sort(key=lambda x: x[1], reverse=True)

        # now Q[0] has the true best marginal node for current S
        v, mg, last = Q.pop(0)

        # add this node to the seed set
        S.append(v)
        current_spread += mg

        # record time
        timeLapse.append(time.time() - startTime)

        print(f"Selected seed {len(S)}: {v}, marginal gain = {mg:.4f}, "
              f"current spread ≈ {current_spread:.4f}")
        print("Time Elapsed: ",time.time()-startTime)
    final_mean_spread = current_spread
    return S, timeLapse, final_mean_spread


import time
import numpy as np

def CELFPP_IC_hypergraph(H, k, p=0.01, mc=10, spread_func=None):
    """
    CELF++ (lazy greedy) for Independent Cascade on a hypergraph.

    Parameters
    ----------
    H : hypernetx.Hypergraph
        Your opinionated hypergraph (nodes = users, hyperedges = opinions).
    k : int
        Number of seeds to select.
    p : float
        Activation probability in IC.
    mc : int
        Number of Monte Carlo simulations for IC.
    spread_func : callable or None
        Function taking a seed list S and returning expected spread.
        If None, uses IC_hypergraph(H, S, p, mc).

    Returns
    -------
    S : list
        Final seed set of size k.
    timeLapse : list of float
        timeLapse[i] = elapsed time (seconds) after selecting (i+1)-th seed.
    final_mean_spread : float
        Expected spread for the final seed set S.
    """

    # Default spread function: IC on your hypergraph
    if spread_func is None:
        def spread_func(S):
            return IC_hypergraph(H, S, p=p, mc=mc)

    start_time = time.time()

    # ----------------------------
    # Data structures for CELF++
    # ----------------------------
    # For each node v we keep:
    #   [v, marginal_gain, last_updated_seed_set_size]
    # This is the "lazy" part; we only recompute when needed.
    Q = []
    S = []                # final seed set
    timeLapse = []
    current_spread = 0.0  # F(S)

    # 1) Initial marginal gain with empty seed set
    print("CELF++: computing initial marginal gains...")
    for v in H.nodes:
        mg = spread_func([v])  # F({v}) - F(∅) = F({v})
        Q.append([v, mg, 0])
        print(Q);

    # sort by marginal_gain descending
    Q.sort(key=lambda x: x[1], reverse=True)

    # ----------------------------
    # 2) CELF++ main loop
    # ----------------------------
    while len(S) < k:
        while True:
            v, mg, last = Q[0]

            # if this marginal gain is already computed for current |S|,
            # we accept it as the true best
            if last == len(S):
                break

            # otherwise recompute gain w.r.t. current S
            new_spread = spread_func(S + [v])
            new_mg = new_spread - current_spread

            # update top element and resort
            Q[0] = [v, new_mg, len(S)]
            Q.sort(key=lambda x: x[1], reverse=True)

        # now Q[0] is the best node with up-to-date marginal gain
        v, mg, last = Q.pop(0)

        # add v to seed set
        S.append(v)
        current_spread += mg
        timeLapse.append(time.time() - start_time)

        print(f"[CELF++] selected seed {len(S)}: {v}, "
              f"marginal gain = {mg:.4f}, spread ≈ {current_spread:.4f}")

    final_mean_spread = current_spread
    return S, timeLapse, final_mean_spread
