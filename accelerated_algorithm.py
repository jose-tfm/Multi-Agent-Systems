import numpy as np
import itertools
import networkx as nx
from scipy.optimize import minimize
from numpy.linalg import eigvals, eig
import matplotlib.pyplot as plt
import io
import pandas as pd
from pandas import DataFrame, ExcelWriter
from openpyxl.drawing.image import Image as OpenPyXLImage

# For neat printing of small arrays
np.set_printoptions(precision=3, suppress=True)

# --- Row‐normalize helper ---
def row_normalize(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

# --- Build augmented P_pa ---
def build_Ap(N: int, A: np.ndarray, optimized: bool = False) -> np.ndarray:
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    for i in range(N):
        base = N + 3*i
        if not optimized:
            Ap[i,     base  ] = 1
            Ap[base,  i     ] = 1
            Ap[i,     base+2] = 2
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
        else:
            Ap[i,     base  ] = 0.186
            Ap[base,  i     ] = 1
            Ap[i,     base+2] = 0.108
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
    return row_normalize(Ap)

# --- Detect first “switch” round ---
def first_full_switch(full: list[float], hist: list[float], eps: float) -> int:
    T = len(full) - 1
    for k in range(T+1):
        if all(abs(hist[t] - full[t]) < eps for t in range(k, T+1)):
            return k
    return T

# --- Validate a minor subgraph ---
def validate_minor(A_sub: np.ndarray) -> bool:
    M = A_sub.copy()
    np.fill_diagonal(M, 0)
    if np.any(M.sum(axis=1) == 0):
        return False
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    return nx.is_strongly_connected(G)

# --- Objective for subgraph optimization ---
def consensus_rate(A_flat: np.ndarray, n: int, mask: np.ndarray) -> float:
    A = row_normalize(A_flat.reshape((n, n)))
    P = build_Ap(n, A, optimized=True)
    w = np.abs(eigvals(P))
    return float(np.sort(w)[-2])

# --- Extract minor adjacency & survivors ---
def minor(A: np.ndarray, F: list[int]) -> tuple[np.ndarray, list[int]]:
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep

# --- Core simulator ---
def simulate_resilient_consensus(
    A: np.ndarray,
    x0_dict: dict[int, float],
    attacker_val: dict[int, callable],
    f: int,
    eps: float,
    T: int,
    optimize_subgraphs: bool = False
) -> tuple[
    dict[str, dict[int, list[float]]],  # histories
    dict[str, dict[int, list[bool]]],   # filter flags
    dict[str, dict[int, int | None]],   # conv_rounds
    dict[str, dict[int, list[float]]],  # full trajectories
    float,                              # honest average
    dict[str, list[np.ndarray]],        # priv_dict
    dict[str, list[np.ndarray]]         # eig_dict
]:
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    # enumerate minors
    F_list = sorted(
        [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
        key=lambda S: (len(S), sorted(S))
    )
    idx0 = F_list.index(frozenset())

    # prepare storage
    labels    = ['orig'] + (['opt'] if optimize_subgraphs else [])
    P_dict    = {lab: [] for lab in labels}
    priv_dict = {lab: [] for lab in labels}
    eig_dict  = {lab: [] for lab in labels}
    surv_idxs = []

    # build each minor & optimize
    for F in F_list:
        surv   = [i-1 for i in agents if i not in F]
        surv_idxs.append(surv)
        A_sub  = row_normalize(A[np.ix_(surv, surv)].copy())
        assert validate_minor(A_sub), f"Minor {F} is not strongly connected"
        n_sub  = len(surv)

        versions = {'orig': A_sub}
        if optimize_subgraphs:
            p0   = A_sub.flatten()
            mask = (p0 > 0).astype(float)
            bounds = [(0,0) if m==0 else (0,1) for m in mask]
            res = minimize(lambda p: consensus_rate(p, n_sub, mask),
                           p0, method='SLSQP', bounds=bounds,
                           options={'ftol':1e-9,'maxiter':500})
            A_opt = row_normalize(res.x.reshape((n_sub, n_sub)))
            assert validate_minor(A_opt), "Optimized minor invalid"
            versions['opt'] = A_opt

        for lab, Asub in versions.items():
            Ppa = build_Ap(n_sub, Asub, optimized=(lab=='opt'))
            P_dict[lab].append(Ppa)

            # left‐eigenvector
            w, V = eig(Ppa.T)
            v = np.abs(np.real(V[:, np.argmin(np.abs(w-1))]))
            v /= v.sum()
            eig_dict[lab].append(v)

            # private initial using α, β, γ
            x_sub = np.array([x0_dict[i+1] for i in surv])
            alpha = np.full(n_sub, 1.0)
            beta  = np.full(n_sub, 1.0)
            gamma = np.full(n_sub, 1.0)

            priv = np.zeros(4 * n_sub)
            for j in range(n_sub):
                a, b, g = alpha[j], beta[j], gamma[j]
                s = a + b + g
                coeff = 4 * x_sub[j] / s
                priv[j] = 0.0
                base = n_sub + 3*j
                priv[base+0] = coeff * a
                priv[base+1] = coeff * b
                priv[base+2] = coeff * g

            # global rescale so that v^T priv = average(x_sub)
            target  = x_sub.mean()
            current = v @ priv
            priv   *= (target / current)

            priv_dict[lab].append(priv)

    # initialize histories, filter flags, conv_rounds, trajectories
    histories   = {lab:{u:[None]*(T+1) for u in agents} for lab in labels}
    filters     = {lab:{u:[False]*(T+1) for u in agents} for lab in labels}
    conv_rounds = {lab:{u:None for u in agents}      for lab in labels}
    X_store     = {lab:[None]*len(F_list)            for lab in labels}

    # simulate (unchanged) …
    for k in range(T+1):
        for lab in labels:
            for i, Fset in enumerate(F_list):
                surv  = surv_idxs[i]; n_sub = len(surv)
                if k == 0:
                    X = np.zeros((4*n_sub, T+1))
                    X[:,0] = priv_dict[lab][i]
                    X_store[lab][i] = X
                else:
                    X = X_store[lab][i]
                    X[:,k] = P_dict[lab][i] @ X[:,k-1]
                    for att, atk in attacker_val.items():
                        if att not in Fset:
                            j = surv.index(att-1)
                            X[n_sub+3*j:n_sub+3*j+3, k] = atk(k-1)
        # collapse & filter mark (unchanged) …
        for lab in labels:
            for u in agents:
                if k == 0:
                    histories[lab][u][0] = x0_dict[u]
                else:
                    if u in attacker_val:
                        val = attacker_val[u](k-1)
                    else:
                        surv0 = surv_idxs[idx0]
                        full = (
                            X_store[lab][idx0][surv0.index(u-1), k]
                            if (u-1) in surv0 else x0_dict[u]
                        )
                        outs = []
                        for j, Fset in enumerate(F_list):
                            surv_j = surv_idxs[j]
                            if Fset and u not in Fset and (u-1) in surv_j:
                                cand = X_store[lab][j][surv_j.index(u-1), k]
                                if abs(cand-full) >= eps:
                                    outs.append(cand)
                        if len(outs) == 1:
                            filters[lab][u][k] = True
                            val = outs[0]
                        else:
                            val = full
                    histories[lab][u][k] = val
                    if conv_rounds[lab][u] is None and abs(val-honest_avg) < eps:
                        conv_rounds[lab][u] = k

    # full trajectories (unchanged) …
    full_trajs = {lab:{} for lab in labels}
    surv0 = surv_idxs[idx0]
    for lab in labels:
        X0 = X_store[lab][idx0]
        for u in agents:
            if (u-1) in surv0:
                full_trajs[lab][u] = list(X0[surv0.index(u-1), :])
            else:
                full_trajs[lab][u] = [x0_dict[u]]*(T+1)

    return histories, filters, conv_rounds, full_trajs, honest_avg, priv_dict, eig_dict

# helper to find first round all honest filtered
def step_subgraph(filter_flags: dict[int,list[bool]], honest: list[int], T:int) -> int | None:
    for k in range(T+1):
        if all(filter_flags[u][k] for u in honest):
            return k
    return None

# --- Batch runner comparing original vs optimized subgraph detection ---
def run_trials(
    N:int=10, p_edge:float=0.8, f:int=1,
    eps:float=0.08, T:int=400,
    seed0:int=4, trials:int=20
) -> DataFrame:
    records = []
    matrix_rows = []

    with ExcelWriter(f'Both_{N}x{N}_{T}_f{f}.xlsx', engine='openpyxl') as writer:
        wb = writer.book
        ws_plots = wb.create_sheet('plots')

        for t in range(trials):
            np.random.seed(seed0 + t)
            seed = seed0 + t

            # generate a random strongly-connected digraph
            while True:
                G = nx.gnp_random_graph(N, p_edge, seed=seed, directed=True)
                if nx.is_strongly_connected(G):
                    break
                seed += 1
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.rand()
            A = row_normalize(nx.to_numpy_array(G, weight='weight'))

            # record full-A rows
            for i in range(N):
                row = {'trial': t, 'version': 'full', 'row': i}
                for j in range(N):
                    row[f'col_{j}'] = A[i, j]
                matrix_rows.append(row)

            x0 = {u: np.random.rand() for u in range(1, N+1)}
            if f == 1:
                attacker = {2: lambda k: 0.8}
            elif f == 2:
                attacker = {2: lambda k: 1 - 2**(-0.3 * (k+1)), 3: lambda k:  1 + 2**(-0.3 * (k+1))}
            else:
                raise ValueError(f"Unsupported number of attackers: f={f}")

            # simulate original vs optimized, capturing priv & eig dicts
            hist_o, filt_o, conv_o, _, avg, priv_o, eig_o = simulate_resilient_consensus(
                A, x0, attacker, f, eps, T, optimize_subgraphs=False
            )
            hist_p, filt_p, conv_p, _, _,    priv_p, eig_p = simulate_resilient_consensus(
                A, x0, attacker, f, eps, T, optimize_subgraphs=True
            )
            honest = [u for u in hist_o['orig'] if u not in attacker]

            # detection rounds
            k_o = step_subgraph(filt_o['orig'], honest, T)
            k_p = step_subgraph(filt_p['opt'], honest, T)
            records.append({'trial': t, 'Orig_subgraph_k': k_o, 'Opt_subgraph_k': k_p})

            # enumerate all minors
            agents = list(range(1, N+1))
            F_list = sorted(
                [frozenset(c) for k2 in range(f+1)
                            for c in itertools.combinations(agents, k2)],
                key=lambda S: (len(S), sorted(S))
            )

            for idx, F in enumerate(F_list):
                surv = [i-1 for i in agents if i not in F]
                A_sub = row_normalize(A[np.ix_(surv, surv)].copy())

                # orig_minor adjacency
                for ii, u in enumerate(surv):
                    row = {
                        'trial': t,
                        'version': 'orig_minor',
                        'faulty_set': tuple(sorted(F)),
                        'row': u
                    }
                    for jj, v in enumerate(surv):
                        row[f'col_{v}'] = A_sub[ii, jj]
                    matrix_rows.append(row)

                # opt_minor adjacency
                p0   = A_sub.flatten()
                mask = (p0 > 0).astype(float)
                bounds = [(0,0) if m==0 else (0,1) for m in mask]
                res = minimize(
                    lambda p: consensus_rate(p, len(surv), mask),
                    p0, method='SLSQP', bounds=bounds,
                    options={'ftol':1e-9, 'maxiter':500}
                )
                A_opt = row_normalize(res.x.reshape((len(surv), len(surv))))
                for ii, u in enumerate(surv):
                    row = {
                        'trial': t,
                        'version': 'opt_minor',
                        'faulty_set': tuple(sorted(F)),
                        'row': u
                    }
                    for jj, v in enumerate(surv):
                        row[f'col_{v}'] = A_opt[ii, jj]
                    matrix_rows.append(row)

                # orig_aug_init & opt_aug_init
                for lab, pdict in (('orig', priv_o), ('opt', priv_p)):
                    row = {
                        'trial': t,
                        'version': f'{lab}_aug_init',
                        'faulty_set': tuple(sorted(F))
                    }
                    x_p0 = pdict[lab][idx]
                    for j, val in enumerate(x_p0):
                        row[f'x_{j}'] = val
                    matrix_rows.append(row)

                # orig_eig_vec & opt_eig_vec
                for lab, edict in (('orig', eig_o), ('opt', eig_p)):
                    row = {
                        'trial': t,
                        'version': f'{lab}_eig_vec',
                        'faulty_set': tuple(sorted(F))
                    }
                    v = edict[lab][idx]
                    for j, val in enumerate(v):
                        row[f'v_{j}'] = val
                    matrix_rows.append(row)

            # per-trial plots with both honest and malicious agents
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            
            # Plot honest agents
            for u in honest:
                ax1.plot(hist_o['orig'][u], '--', label=f'Honest x_{u}', alpha=0.8)
                ax2.plot(hist_p['opt'][u],  '-', label=f'Honest x_{u}', alpha=0.8)
            
            # Plot malicious agents with different styling
            for u in attacker.keys():
                ax1.plot(hist_o['orig'][u], 'x-', linewidth=2, markersize=4, 
                        label=f'Malicious x_{u}', color='red', alpha=0.9)
                ax2.plot(hist_p['opt'][u], 'x-', linewidth=2, markersize=4, 
                        label=f'Malicious x_{u}', color='red', alpha=0.9)
            
            for ax, title in zip((ax1, ax2), ('Original', 'Optimized')):
                ax.axhline(avg, ls=':', c='k', label='Honest Average')
                ax.set_xlabel('k')
                ax.set_ylabel('Value')
                ax.set_title(f'{title} Trial {t}')
                ax.grid(True, alpha=0.3)
            
            # Add legends with better positioning
            ax1.legend(ncol=2, fontsize='small', loc='best')
            ax2.legend(ncol=2, fontsize='small', loc='best')
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            ws_plots.add_image(OpenPyXLImage(buf), f'A{2 + t*20}')

        # summary sheet
        df = DataFrame(records)
        df.to_excel(writer, sheet_name='summary', index=False)

        # boxplot sheet
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df['Orig_subgraph_k'].dropna(), df['Opt_subgraph_k'].dropna()]
        ax.boxplot(data, labels=['Orig', 'Opt'])
        ax.set_title('Subgraph Detection Steps')
        ax.set_ylabel('k')
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        ws2 = wb.create_sheet('boxplot')
        ws2.add_image(OpenPyXLImage(buf), 'A1')

        # write all_matrices
        mat_df = DataFrame(matrix_rows)
        mat_df.to_excel(writer, sheet_name='all_matrices', index=False)

    print(f"Saved 'Both_{N}x{N}_{T}_f{f}.xlsx' with detection results, plots, and all matrix rows.")
    return DataFrame(records)

if __name__ == '__main__':

    df = run_trials(N=11, p_edge=0.8, f=1, eps=0.05, T=400, seed0=4, trials=1)
    print(df.describe())
