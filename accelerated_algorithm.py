"""
Accelerating Average Consensus in Dynamical Networks 
Without Compromising Accuracy, Privacy or Resilience

This implements the accelerated consensus algorithm described in the thesis,
based on the paper “Accelerating Average Consensus in Dynamical Networks Without 
Compromising Accuracy, Privacy or Resilience.”

Author: José Moniz
Date: October 2025
"""


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

# Configure numpy for readable output
np.set_printoptions(precision=3, suppress=True)


# ============================================================================
# Core Matrix Operations
# ============================================================================

def row_normalize(M: np.ndarray) -> np.ndarray:
    """
    Row-normalize a matrix to obtain a row-stochastic matrix.
    
    Args:
        M: Input matrix (N x N)
        
    Returns:
        Row-stochastic matrix where each row sums to 1
        
    Note:
        Adds small epsilon (1e-12) to prevent division by zero.
        Rows with zero sum remain zero (handled by connectivity checks).
    """
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)


def build_Ap(N: int, A: np.ndarray, optimized: bool = False) -> np.ndarray:
    """
    Construct the augmented private-accumulator consensus matrix P_pa.
    
    This builds a 4N×4N matrix with 3 private accumulator states (α, β, γ) 
    per agent for Byzantine-resilient consensus.
    
    Args:
        N: Number of agents
        A: Row-normalized adjacency matrix (N × N)
        optimized: If True, use tuned cross-couplings for faster convergence
        
    Returns:
        Augmented consensus matrix P_pa of size 4N × 4N
        
    References:
        Standard configuration uses unit weights for accumulator coupling.
        Optimized configuration uses weights (0.186, 0.108) derived from 
        spectral optimization to minimize the second-largest eigenvalue modulus.
    """
    size = 4 * N
    Ap = np.zeros((size, size))
    Ap[:N, :N] = A
    
    for i in range(N):
        base = N + 3*i
        if not optimized:
            # Standard configuration: unit weights
            Ap[i,     base  ] = 1
            Ap[base,  i     ] = 1
            Ap[i,     base+2] = 2
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
        else:
            # Optimized configuration: tuned for faster convergence
            Ap[i,     base  ] = 0.186
            Ap[base,  i     ] = 1
            Ap[i,     base+2] = 0.108
            Ap[base+2, base+1] = 1
            Ap[base+1, i     ] = 1
    
    return row_normalize(Ap)


# ============================================================================
# Byzantine Resilience & Detection
# ============================================================================

def first_full_switch(full: list[float], hist: list[float], eps: float) -> int:
    """
    Detect the first round where filtered trajectory permanently matches full trajectory.
    
    Args:
        full: Unfiltered trajectory values
        hist: Filtered trajectory values
        eps: Detection threshold
        
    Returns:
        First time step k where |hist[t] - full[t]| < eps for all t >= k
    """
    T = len(full) - 1
    for k in range(T+1):
        if all(abs(hist[t] - full[t]) < eps for t in range(k, T+1)):
            return k
    return T


def validate_minor(A_sub: np.ndarray) -> bool:
    """
    Validate that a minor subgraph is strongly connected.
    
    For Byzantine-resilient consensus, each minor (after removing up to f agents)
    must remain strongly connected to guarantee convergence.
    
    Args:
        A_sub: Adjacency matrix of the minor subgraph
        
    Returns:
        True if strongly connected, False otherwise
    """
    M = A_sub.copy()
    np.fill_diagonal(M, 0)
    # Quick out-degree check
    if np.any(M.sum(axis=1) == 0):
        return False
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    return nx.is_strongly_connected(G)


# ============================================================================
# Spectral Optimization
# ============================================================================

def consensus_rate(A_flat: np.ndarray, n: int, mask: np.ndarray) -> float:
    """
    Compute consensus convergence rate using second-largest eigenvalue modulus (SLEM).
    
    This objective function is minimized during subgraph optimization to accelerate
    consensus convergence by reducing the spectral gap.
    
    Args:
        A_flat: Flattened adjacency matrix weights (n² elements)
        n: Number of agents in subgraph
        mask: Binary mask indicating allowed edges (fixed support)
        
    Returns:
        SLEM of augmented matrix P_pa (smaller is faster convergence)
        
    Note:
        The mask parameter is used externally to set optimization bounds.
        Zero entries in mask correspond to (0,0) bounds in the optimizer.
    """
    A = row_normalize(A_flat.reshape((n, n)))
    P = build_Ap(n, A, optimized=True)
    w = np.abs(eigvals(P))
    return float(np.sort(w)[-2])


def minor(A: np.ndarray, F: list[int]) -> tuple[np.ndarray, list[int]]:
    """
    Extract minor subgraph by removing faulty agents.
    
    Args:
        A: Full adjacency matrix (N × N)
        F: List of faulty agent indices (0-based)
        
    Returns:
        Tuple of (submatrix, list of surviving agent indices)
    """
    keep = [i for i in range(A.shape[0]) if i not in F]
    return A[np.ix_(keep, keep)], keep


# ============================================================================
# Main Simulation Algorithm
# ============================================================================

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
    """
    Simulate Byzantine-resilient consensus with accelerated subgraph-based detection.
    
    This implements a multi-agent consensus algorithm resilient to Byzantine faults,
    using private accumulator variables (α, β, γ) and optimized minor subgraph detection
    to accelerate identification and filtering of malicious agents.
    
    Args:
        A: Communication adjacency matrix (N × N), will be row-normalized internally
        x0_dict: Initial values {agent_id (1-indexed): value}
        attacker_val: Byzantine behaviors {agent_id: value_function(k)}
                     Function should return scalar (broadcast to 3 accumulators)
                     or array of shape (3,) for direct accumulator injection
        f: Maximum number of Byzantine agents tolerated
        eps: Convergence/detection threshold
        T: Number of simulation time steps (0 to T inclusive)
        optimize_subgraphs: If True, optimize minor subgraph weights via SLSQP
                           to minimize SLEM (accelerates detection)
    
    Returns:
        histories: Filtered agent trajectories {label: {agent_id: [values]}}
        filters: Subgraph filter activation flags {label: {agent_id: [bool]}}
        conv_rounds: First convergence round {label: {agent_id: k or None}}
        full_trajs: Unfiltered trajectories from full graph
        honest_avg: Average of initial honest agent values
        priv_dict: Private accumulator initializations per minor
        eig_dict: Left eigenvectors (stationary distributions) per minor
        
    Labels:
        'orig': Original (unoptimized) subgraph weights
        'opt': Optimized subgraph weights (if optimize_subgraphs=True)
    
    Algorithm Overview:
        1. Enumerate all possible faulty sets F with |F| ≤ f
        2. For each F, construct minor subgraph and validate strong connectivity
        3. Optionally optimize minor weights to minimize spectral radius
        4. Initialize private accumulators (α=β=γ=1) with proper scaling
        5. Simulate consensus dynamics on all minors in parallel
        6. Apply subgraph detection: if unique outlier detected, filter to that minor
        7. Track convergence to honest average and filter activation times
    """
    N = A.shape[0]
    agents = list(range(1, N+1))
    honest_avg = np.mean([x0_dict[u] for u in agents if u not in attacker_val])

    # Enumerate all possible faulty sets F with |F| ≤ f
    F_list = sorted(
        [frozenset(c) for k in range(f+1) for c in itertools.combinations(agents, k)],
        key=lambda S: (len(S), sorted(S))
    )
    idx0 = F_list.index(frozenset())

    # Prepare storage for each version (original and optionally optimized)
    labels    = ['orig'] + (['opt'] if optimize_subgraphs else [])
    P_dict    = {lab: [] for lab in labels}
    priv_dict = {lab: [] for lab in labels}
    eig_dict  = {lab: [] for lab in labels}
    surv_idxs = []

    # Build each minor subgraph and optimize weights if requested
    for F in F_list:
        surv   = [i-1 for i in agents if i not in F]
        surv_idxs.append(surv)
        A_sub  = row_normalize(A[np.ix_(surv, surv)].copy())
        assert validate_minor(A_sub), f"Minor {F} is not strongly connected"
        n_sub  = len(surv)

        # Start with original adjacency
        versions = {'orig': A_sub}
        
        if optimize_subgraphs:
            # Optimize subgraph weights via SLSQP to minimize SLEM
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
            # Build augmented consensus matrix P_pa
            Ppa = build_Ap(n_sub, Asub, optimized=(lab=='opt'))
            P_dict[lab].append(Ppa)

            # Compute left eigenvector (stationary distribution)
            w, V = eig(Ppa.T)
            v = np.abs(np.real(V[:, np.argmin(np.abs(w-1))]))
            v /= v.sum()
            eig_dict[lab].append(v)

            # Initialize private accumulators: α = β = γ = 1
            x_sub = np.array([x0_dict[i+1] for i in surv])
            alpha = np.full(n_sub, 1.0)
            beta  = np.full(n_sub, 1.0)
            gamma = np.full(n_sub, 1.0)

            priv = np.zeros(4 * n_sub)
            for j in range(n_sub):
                a, b, g = alpha[j], beta[j], gamma[j]
                s = a + b + g
                coeff = 4 * x_sub[j] / s
                priv[j] = 0.0  # Public state starts at 0
                base = n_sub + 3*j
                priv[base+0] = coeff * a  # α accumulator
                priv[base+1] = coeff * b  # β accumulator
                priv[base+2] = coeff * g  # γ accumulator

            # Global rescale so that v^T · priv = mean(x_sub)
            target  = x_sub.mean()
            current = v @ priv
            priv   *= (target / current)

            priv_dict[lab].append(priv)

    # Initialize trajectory storage
    histories   = {lab:{u:[None]*(T+1) for u in agents} for lab in labels}
    filters     = {lab:{u:[False]*(T+1) for u in agents} for lab in labels}
    conv_rounds = {lab:{u:None for u in agents}      for lab in labels}
    X_store     = {lab:[None]*len(F_list)            for lab in labels}

    # Main simulation loop: iterate consensus dynamics on all minors
    for k in range(T+1):
        for lab in labels:
            for i, Fset in enumerate(F_list):
                surv  = surv_idxs[i]
                n_sub = len(surv)
                
                if k == 0:
                    # Initialize state trajectory
                    X = np.zeros((4*n_sub, T+1))
                    X[:,0] = priv_dict[lab][i]
                    X_store[lab][i] = X
                else:
                    # Consensus update: X[k] = P · X[k-1]
                    X = X_store[lab][i]
                    X[:,k] = P_dict[lab][i] @ X[:,k-1]
                    
                    # Byzantine agents inject malicious values into accumulators
                    for att, atk in attacker_val.items():
                        if att not in Fset:
                            j = surv.index(att-1)
                            X[n_sub+3*j:n_sub+3*j+3, k] = atk(k-1)
        
        # Collapse states and apply subgraph filtering
        for lab in labels:
            for u in agents:
                if k == 0:
                    histories[lab][u][0] = x0_dict[u]
                else:
                    if u in attacker_val:
                        # Attackers display their injected values
                        val = attacker_val[u](k-1)
                    else:
                        # Honest agents: get full-graph value
                        surv0 = surv_idxs[idx0]
                        full = (
                            X_store[lab][idx0][surv0.index(u-1), k]
                            if (u-1) in surv0 else x0_dict[u]
                        )
                        
                        # Check for unique outlier among minors (subgraph detection)
                        outs = []
                        for j, Fset in enumerate(F_list):
                            surv_j = surv_idxs[j]
                            if Fset and u not in Fset and (u-1) in surv_j:
                                cand = X_store[lab][j][surv_j.index(u-1), k]
                                if abs(cand-full) >= eps:
                                    outs.append(cand)
                        
                        # If exactly one outlier detected, filter to that minor
                        if len(outs) == 1:
                            filters[lab][u][k] = True
                            val = outs[0]
                        else:
                            val = full
                    
                    histories[lab][u][k] = val
                    
                    # Track first convergence to honest average
                    if conv_rounds[lab][u] is None and abs(val-honest_avg) < eps:
                        conv_rounds[lab][u] = k

    # Compute full (unfiltered) trajectories for reference
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


# ============================================================================
# Experimental Utilities
# ============================================================================

def step_subgraph(filter_flags: dict[int,list[bool]], honest: list[int], T:int) -> int | None:
    """
    Find first round where all honest agents have activated subgraph filtering.
    
    Args:
        filter_flags: Filter activation history per agent
        honest: List of honest agent IDs
        T: Maximum time steps
        
    Returns:
        First time k where all honest agents are filtered, or None if never
    """
    for k in range(T+1):
        if all(filter_flags[u][k] for u in honest):
            return k
    return None


# ============================================================================
# Batch Experimental Runner
# ============================================================================

def run_trials(
    N:int=10, p_edge:float=0.8, f:int=1,
    eps:float=0.08, T:int=400,
    seed0:int=4, trials:int=20
) -> DataFrame:
    """
    Run batch experiments comparing original vs optimized subgraph detection.
    
    Generates random strongly-connected digraphs, simulates Byzantine consensus
    with both standard and optimized configurations, and exports results including 
    detection times, trajectories, matrices, and visualizations.
    
    Args:
        N: Number of agents
        p_edge: Edge probability for random graph generation
        f: Number of Byzantine agents (currently supports f=1 or f=2)
        eps: Detection/convergence threshold
        T: Simulation time horizon
        seed0: Initial random seed
        trials: Number of independent trials to run
        
    Returns:
        DataFrame with detection round comparisons per trial
        
    Outputs:
        Excel file 'Both_{N}x{N}_{T}_f{f}.xlsx' containing:
        - summary: Detection times for original vs optimized
        - plots: Trajectory plots for honest and malicious agents
        - boxplot: Statistical comparison of detection performance
        - all_matrices: Full/minor adjacencies, initial states, eigenvectors
        
    Byzantine Behaviors:
        f=1: Constant value attack (agent 2 → 0.8)
        f=2: Exponential approach attacks (agents 2,3 → 1±2^(-0.3k))
    """
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
                attacker = {2: lambda k: 0.8, 3: lambda k:  1 + 2**(-0.3 * (k+1))}
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


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Run single trial with N=11 agents, f=1 Byzantine agent
    df = run_trials(N=8, p_edge=0.8, f=2, eps=0.15, T=400, seed0=4, trials=1)
    print(df.describe())
