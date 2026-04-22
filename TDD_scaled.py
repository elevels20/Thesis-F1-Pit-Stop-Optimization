# NO APPENDIX G!!!
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import linprog
import math

# PARAMETERS
N = 10 # Number of laps 
# T = [1, 2, 3] # Tire compounds. 1 = Soft, 2 = Medium, 3 = Hard
T = [1, 2] 
T_cartesian = list(itertools.product(T, T))
T0 = [0] + T # 0 = no pit stop

# u1 = 30 # Lifespan in number of laps for Soft tires
# u2 = 40 # Lifespan in number of laps for Medium tires
# u3 = 50 # Lifespan in number of laps for Hard tires
u1 = 5
u2 = 7
# u1 = 3
# u2 = 4
# u3 = 5
u = [u1, u2]

d0 = {"A":97.22, "B":97.24, "C": 97.20} # Lap time for driver A/B/C when using new soft compound tires at the beginning of the race (without pit stops, interactions, or DRS)
p0 = {"A":20.2, "B":20.0, "C": 20.4} # Additional lap time for driver A/B/C due to a pit stop

lambda_pen = 2.0
h = 0.02 # Lap time reduction between two consecutive laps attributed to fuel consumption (making the car lighter)

p_VSC = {"A":10.0, "B":10.0, "C": 10.0} # Additional lap time for driver A/B/C due to a pit stop under VSC
p_SC  = {"A":10.0, "B":10.0, "C": 10.0} #additional lap time for driver A/B/C due to a pit stop under SC

g_AB1 = -0.4 # Initial time gap between A and B
g_AC1 = -0.8 # Initial time gap between A and C

# gap discretization
g_min = -2
g_max = 2
g_step = 0.4  
g_values = np.arange(g_min, g_max + g_step, g_step)

# Scaling
SCALE = 2 / 35

p0 = {k: v * SCALE for k, v in p0.items()}
p_SC = {k: v * SCALE for k, v in p_SC.items()}
p_VSC = {k: v * SCALE for k, v in p_VSC.items()}
h *= SCALE
lambda_pen = 2.0 / SCALE

# Fix baseline lap times
d0 = {k: v * SCALE for k, v in d0.items()}

z1 = 0.4
z2 = 0.7

t_drs = 0.3

# state = (tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC)

# FUNCTIONS

# Tire-wear function
def tire_wear(tire, w):
    if tire == 1:
        return SCALE * (0.004 * (w - 1) + 0.005 * ((w - 1) ** 2))
    elif tire == 2:
        return SCALE * (0.4 + 0.20 * w + 0.008 * (w ** 2))
    elif tire == 3:
        return SCALE * (0.9 + 0.10 * w + 0.0001 * (w ** 2))

# Interaction function
def interaction_A(eps_AB, eps_AC):
    interaction_AB = 0
    interaction_AC = 0

    if eps_AB < 0:
        interaction_AB = np.exp(-lambda_pen * abs(eps_AB)) * z1
    else:
        interaction_AB = np.exp(-lambda_pen * abs(eps_AB)) * z2

    if eps_AC < 0:
        interaction_AC = np.exp(-lambda_pen * abs(eps_AC)) * z1
    else:
        interaction_AC = np.exp(-lambda_pen * abs(eps_AC)) * z2

    return interaction_AB + interaction_AC

def interaction_B(eps_AB, eps_BC):
    interaction_BA = 0
    interaction_BC = 0

    if eps_AB >= 0:
        interaction_BA = np.exp(-lambda_pen * abs(eps_AB)) * z1
    else:
        interaction_BA = np.exp(-lambda_pen * abs(eps_AB)) * z2

    if eps_BC < 0:
        interaction_BC = np.exp(-lambda_pen * abs(eps_BC)) * z1
    else:
        interaction_BC = np.exp(-lambda_pen * abs(eps_BC)) * z2

    return interaction_BA + interaction_BC

def interaction_C(eps_AC, eps_BC):
    interaction_CA = 0
    interaction_CB = 0

    if eps_AC >= 0:
        interaction_CA = np.exp(-lambda_pen * abs(eps_AC)) * z1
    else:
        interaction_CA = np.exp(-lambda_pen * abs(eps_AC)) * z2

    if eps_BC >= 0:
        interaction_CB = np.exp(-lambda_pen * abs(eps_BC)) * z1
    else:
        interaction_CB = np.exp(-lambda_pen * abs(eps_BC)) * z2

    return interaction_CA + interaction_CB

# Lap time functions

def driver_order(g_AB, g_AC, g_BC, pitA, pitB, pitC):
    IA = 1 if pitA else 0
    IB = 1 if pitB else 0
    IC = 1 if pitC else 0

    xi_AB = 1 if (g_AB + p_SC["A"] * IA - p_SC["B"] * IB) < 0 else 0  # 1 if driver A ends up ahead of B after the pit stop exit of a particular lap
    xi_AC = 1 if (g_AC + p_SC["A"] * IA - p_SC["C"] * IC) < 0 else 0  # 1 if driver A ends up ahead of C after the pit stop exit of a particular lap
    xi_BC = 1 if (g_BC + p_SC["B"] * IB - p_SC["C"] * IC) < 0 else 0  # 1 if driver B ends up ahead of C after the pit stop exit of a particular lap

    if xi_AB == 1:
        if xi_AC == 1:
            P1 = "A"
            if xi_BC == 1:
                P2 = "B"
                P3 = "C"
            else:
                P2 = "C"
                P3 = "B"
        else:
            if xi_BC == 1:
                raise ValueError("Order of drivers not possible")
            else:
                P1 = "C"
                P2 = "A"
                P3 = "B"
    else:
        if xi_BC == 1:
            P1 = "B"
            if xi_AC == 1:
                P2 = "A"
                P3 = "C"
            else:
                P2 = "C"
                P3 = "A"
        else:
            if xi_AC == 1:
                raise ValueError("Order of drivers not possible")
            else:
                P1 = "C"
                P2 = "B"
                P3 = "A"

    return (P1, P2, P3)

def lap_time(driver, n, tire_n, w, pitA, pitB, pitC, g_AB, g_AC, g_BC): 
    IA = 1 if pitA else 0
    IB = 1 if pitB else 0
    IC = 1 if pitC else 0

    drs_AB = 1 if (0 <= g_AB <= 1) else 0
    drs_BA = 1 if (-1 <= g_AB <= 0) else 0
    drs_AC = 1 if (0 <= g_AC <= 1) else 0
    drs_CA = 1 if (-1 <= g_AC <= 0) else 0
    drs_BC = 1 if (0 <= g_BC <= 1) else 0
    drs_CB = 1 if (-1 <= g_AB <= 0) else 0

    eps_AB = g_AB + p0["A"] * IA - t_drs * drs_AB - (p0["B"] * IB - t_drs * drs_BA)
    eps_AC = g_AC + p0["A"] * IA - t_drs * drs_AC - (p0["C"] * IC - t_drs * drs_CA)
    eps_BC = g_BC + p0["B"] * IB - t_drs * drs_BC - (p0["C"] * IC - t_drs * drs_CB)

    if driver == "A":
        eta = interaction_A(eps_AB, eps_AC)
        pit = pitA
        drs = 1 if (0 <= g_AB <= 1 or 0 <= g_AC <= 1) else 0
    elif driver == "B":
        eta = interaction_B(eps_AB, eps_BC)
        pit = pitB
        drs = 1 if (-1 <= g_AB <= 0 or 0 <= g_BC <= 1) else 0
    else:
        eta = interaction_C(eps_AC, eps_BC)
        pit = pitC
        drs = 1 if (-1 <= g_AC <= 0 or -1 <= g_BC <= 0) else 0

    gamma = d0[driver] + (p0[driver] if pit else 0) + tire_wear(tire_n, w) - h * n
    
    return gamma + eta - t_drs * drs
    
# State functions
def t_next(tire, decision):
    if decision == 0:
        return tire
    else:
        return decision
    
def w_next(w, decision, u):
    I_driver = 1 if decision == 0 else 0

    return min(w * I_driver + 1, u + 1)

def m_next(m, decision, tire):
    if decision != 0 and decision != tire:
        return max(m, 1)
    else:
        return max(m, 0)
    
def g_AB_next(n, tire_A_n, tire_B_n, wA, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC):
    return g_AB + lap_time("A", n, tire_A_n, wA, pitA, pitB, pitC, g_AB, g_AC, g_BC) - lap_time("B", n, tire_B_n, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC)
    
def g_AC_next(n, tire_A_n, tire_C_n, wA, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC):
    return g_AC + lap_time("A", n, tire_A_n, wA, pitA, pitB, pitC, g_AB, g_AC, g_BC) - lap_time("C", n, tire_C_n, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC)
    
def g_BC_next(g_AB_n, g_AC_n):
    return g_AC_n - g_AB_n
    
def discretize_gap(g):
    return min(g_values, key=lambda x: abs(x - g))
    
def state_next(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC, n, decisionA, decisionB, decisionC):
    tire_A_n = t_next(tire_A, decisionA)
    tire_B_n = t_next(tire_B, decisionB)
    tire_C_n = t_next(tire_C, decisionC)

    wA_n = w_next(wA, decisionA, u[tire_A - 1])
    wB_n = w_next(wB, decisionB, u[tire_B - 1])
    wC_n = w_next(wC, decisionC, u[tire_C - 1])

    mA_n = m_next(mA, decisionA, tire_A)
    mB_n = m_next(mB, decisionB, tire_B)
    mC_n = m_next(mC, decisionC, tire_C)

    pitA = True if decisionA != 0 else False
    pitB = True if decisionB != 0 else False
    pitC = True if decisionC != 0 else False

    g_BC = g_AC - g_AB

    g_AB_n = g_AB_next(n, tire_A_n, tire_B_n, wA, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC)
    g_AC_n = g_AC_next(n, tire_A_n, tire_C_n, wA, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC)
    g_BC_n = g_BC_next(g_AB_n, g_AC_n)
    
    g_AB_n = discretize_gap(g_AB_n)
    g_AC_n = discretize_gap(g_AC_n)
    g_BC_n = discretize_gap(g_BC_n)

    return (tire_A_n, wA_n, mA_n, tire_B_n, wB_n, mB_n, tire_C_n, wC_n, mC_n, g_AB_n, g_AC_n)


# Stochastic dynamic programming
def H(w, u, m): 
    """
    Returns whether the driver finishes the race adequately without using tires 
    longer than their lifespans and using at least two different tire compounds.
    """
    return w <= u and m == 1

def V_end(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC):
    Ha = H(wA, u[tire_A - 1], mA)
    Hb = H(wB, u[tire_B - 1], mB)
    Hc = H(wC, u[tire_C - 1], mC)

    if Ha:
        if Hb:
            if Hc:
                return max(g_AB, g_AC)
            else:
                return g_AB
        else:
            if Hc:
                return g_AC
            else:
                return - math.inf
    elif (not Ha) and (not Hb) and (not Hc):
        return 0
    else:
        return math.inf

def V_end_new(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC):
    Ha = H(wA, u[tire_A - 1], mA)
    Hb = H(wB, u[tire_B - 1], mB)
    Hc = H(wC, u[tire_C - 1], mC)

    if Ha:
        if Hb and Hc:
            return max(g_AB, g_AC)
        else:
                return - math.inf
    elif (not Ha) and (not Hb) and (not Hc):
        return 0
    else:
        return math.inf

# state = (tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC)

def generate_states(n):
    states = []

    for tire_A in T:
        for tire_B in T:
            for tire_C in T:

                for wA in range(u[tire_A - 1] + 2):
                    if wA > n: # tire wear cannot be higher than current lap
                        continue
                    for wB in range(u[tire_B - 1] + 2):
                        if wB > n: # tire wear cannot be higher than current lap
                            continue
                        for wC in range(u[tire_C - 1] + 2):
                            if wC > n: # tire wear cannot be higher than current lap
                                continue

                            for mA in range(2):
                                if mA == 1 and n == 1: # mA = 1 only if pit happened in a previous lap, this cannot hapen in the first lap
                                    continue
                                for mB in range(2):
                                    if mB == 1 and n == 1: # mB = 1 only if pit happened in a previous lap, this cannot hapen in the first lap
                                        continue
                                    for mC in range(2):
                                        if mC == 1 and n == 1: # mC = 1 only if pit happened in a previous lap, this cannot hapen in the first lap
                                            continue
                                    
                                        for g_AB in g_values:
                                            for g_AC in g_values:

                                                states.append((
                                                    tire_A, wA, mA,
                                                    tire_B, wB, mB,
                                                    tire_C, wC, mC,
                                                    g_AB, g_AC
                                                ))

    return states

# Stochastic Dynamic Programming Algorithm
def solve_SDP():
    # Step 3: Compute 𝑉 ′^{*} _N+1(s_n) for all s_n ∈ S_N+1
    states_final = generate_states(N + 1) 
    V = {state: 0 for state in states_final}
    for state in states_final:
        tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC = state
        # V[state] = V_end(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC)
        V[state] = V_end_new(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC)

    # Policies
    xA_star = {}
    xBC_star = {}

    for n in range(N, 0, -1):
        if n == 1 or n == N:  # No pit in lap 1 or N
            T_allowed = [0]
        else:
            T_allowed = T0

        T_allowed_cartesian = list(itertools.product(T_allowed, T_allowed))

        V_new = {}
        states = generate_states(n)
        for state in states:
            tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC = state

            # Step 6: compute V'_n(s,a,(b,c)) for all a ∈ T_allowed, (b, c) ∈ T_allowed^2
            V_prime = {}
            for a in T_allowed:
                for b,c in T_allowed_cartesian:
                    val = 0
                    state_n = state_next(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC, n, a, b, c)

                    val += V[state_n]                                    

                    V_prime[(a, (b, c))] = val

            if g_AB < 0 and g_AC < 0: # A is leader (minimize g)
                # Step 8: compute (x_n^{B*}, x_n^{C*})(s,a) for all a ∈ T_allowed.
                bc_star = {a: max(T_allowed_cartesian, key=lambda bc: V_prime[(a, bc)]) for a in T_allowed}

                # Step 9: compute x_n^{A*}(s)
                a_star = min(T_allowed, key=lambda a: V_prime[(a, bc_star[a])])

                # Step 10: value update
                V_new[state] = V_prime[(a_star, bc_star[a_star])]

                xA_star[(n, state)] = a_star
                xBC_star[(n, state)] = bc_star[a_star]
            
            else: # A is follower 
                # Step 12: compute x_n^{A*}(s,(b, c)) for all b, c ∈ T_allowed.
                a_star = {bc: min(T_allowed, key=lambda a: V_prime[(a, bc)]) for bc in T_allowed_cartesian}

                # Step 13: compute (x_n^{B*}, x_n^{C*})(s)
                bc_star = max(T_allowed_cartesian, key=lambda bc: V_prime[(a_star[bc], bc)])

                # Step 14: value update
                V_new[state] = V_prime[(a_star[bc_star], bc_star)]

                xA_star[(n, state)] = a_star[bc_star]
                xBC_star[(n, state)] = bc_star
    
        V = V_new


    nT = len(T)
    nT_cartesian = len(T_cartesian)

    # Step 15: build payoff matrix U'
    U = np.zeros((nT, nT_cartesian))

    g_AB_init = discretize_gap(g_AB1)
    g_AC_init = discretize_gap(g_AC1)

    for i, tA in enumerate(T):
        for j, (tB, tC) in enumerate(T_cartesian):
            s1 = (tA, 0, 0, tB, 0, 0, tC, 0, 0, g_AB_init, g_AC_init)
            U[i, j] = V[s1]

    # Step 16: solve zero-sum game via LP   
    # Players (B, C) (E.3)

    # Variables: [π_BC (nT_cartesian), ρ]
    c = np.zeros(nT_cartesian + 1) # Coefficients of the objective function to minimize.
    c[-1] = -1  # maximize ρ

    A_ub = [] # Coefficients for the inequality constraints 
    b_ub = [] # Right-hand side values for the inequality constraints.

    # ρ - (U π_BC)_i ≤ 0  →  -U[i,:] π_BC + ρ ≤ 0
    for i in range(nT):
        A_ub.append([-U[i, j] for j in range(nT_cartesian)] + [1])
        b_ub.append(0)

    A_eq = [[1]*nT_cartesian + [0]] # Coefficients for the equality constraints (Ax = b).
    b_eq = [1] # Right-hand side value for the equality constraints.

    bounds = [(0,1)]*nT_cartesian + [(None, None)] # Specifies the bounds for each variable in the form of a sequence of tuples.

    resB = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    pi_BC = resB.x[:-1]

    # Player A (E.4)
    # Variables: [π_A (nT), ρ]
    c = np.zeros(nT + 1)
    c[-1] = 1  # minimize ρ

    A_ub = []
    b_ub = []

    # ρ - (U^T π_A)_i ≥ 0  →  (U^T π_A)_i - ρ ≤ 0
    for i in range(nT_cartesian):
        A_ub.append([U[j, i] for j in range(nT)] + [-1])
        b_ub.append(0)

    A_eq = [[1]*nT + [0]]
    b_eq = [1]

    bounds = [(0,1)]*nT + [(None, None)]

    resA = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    pi_A = resA.x[:-1]

    return U, pi_A, pi_BC, xA_star, xBC_star


def simulate_race(pi_A, pi_BC, xA_star, xBC_star):
    # Initial state
    tA = np.random.choice(T, p=pi_A)

    idx_BC = np.random.choice(len(T_cartesian), p=pi_BC)
    tB, tC = T_cartesian[idx_BC]

    g_AB = discretize_gap(g_AB1)
    g_AC = discretize_gap(g_AC1)

    state = (tA, 0, 0, tB, 0, 0, tC, 0, 0, g_AB, g_AC)

    history = []
    gap_history = []
    pit_history = []

    # Simulate laps
    for n in range(1, N + 1):

        tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC = state

        # Optimal decisions from SDP
        a = xA_star[(n, state)]
        b, c = xBC_star[(n, state)]

        pitA = (a != 0)
        pitB = (b != 0)
        pitC = (c != 0)

        history.append((n, tire_A, tire_B, tire_C))
        gap_history.append((g_AB, g_AC))
        pit_history.append((pitA, pitB, pitC, a, b, c))

        # Compute next state
        state = state_next(
            tire_A, wA, mA,
            tire_B, wB, mB,
            tire_C, wC, mC, 
            g_AB, g_AC,
            n,
            a, b, c
        )

    # Other way: 
    tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC = state
    val = V_end_new(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC)

    if val > 0:
        winner = "BC"
    elif val < 0:
        winner = "A"  
    else:
        winner = "tie"
 

    return (g_AB, g_AC), winner, history, gap_history, pit_history

def run_simulations(U, pi_A, pi_BC, xA_star, xBC_star, n_sim=10000):
    results = []

    for _ in range(n_sim):
        g_final, winner, _, _, _ = simulate_race(pi_A, pi_BC, xA_star, xBC_star)
        results.append((g_final, winner))

    g_AB = np.array([r[0][0] for r in results])
    g_AC = np.array([r[0][1] for r in results])
    winners = [r[1] for r in results]

    # Probabilities
    p_A_win = np.mean([w == "A" for w in winners])
    p_B_win = np.mean([w == "B" for w in winners])
    p_C_win = np.mean([w == "C" for w in winners])
    p_BC_win = np.mean([w == "BC" for w in winners])

    # Unconditional stats
    mean_g_AB = np.mean(g_AB)
    mean_g_AC = np.mean(g_AC)
    std_g_AB = np.std(g_AB)
    std_g_AC = np.std(g_AC)

    # Conditional stats for A wins
    gaps_AB_A_win = [g_AB[i] for i, w in enumerate(winners) if w == "A"]
    gaps_AC_A_win = [g_AC[i] for i, w in enumerate(winners) if w == "A"]

    # Conditional stats for B wins
    gaps_AB_B_win = [g_AB[i] for i, w in enumerate(winners) if w == "B"]
    gaps_AC_B_win = [g_AC[i] for i, w in enumerate(winners) if w == "B"]

    # Conditional stats for C wins
    gaps_AB_C_win = [g_AB[i] for i, w in enumerate(winners) if w == "C"]
    gaps_AC_C_win = [g_AC[i] for i, w in enumerate(winners) if w == "C"]

    # Conditional stats for BC wins
    gaps_AB_BC_win = [g_AB[i] for i, w in enumerate(winners) if w == "BC"]
    gaps_AC_BC_win = [g_AC[i] for i, w in enumerate(winners) if w == "BC"]

    # Means
    mean_AB_A_win = np.mean(gaps_AB_A_win) if gaps_AB_A_win else 0
    mean_AC_A_win = np.mean(gaps_AC_A_win) if gaps_AC_A_win else 0

    mean_AB_B_win = np.mean(gaps_AB_B_win) if gaps_AB_B_win else 0
    mean_AC_B_win = np.mean(gaps_AC_B_win) if gaps_AC_B_win else 0

    mean_AB_C_win = np.mean(gaps_AB_C_win) if gaps_AB_C_win else 0
    mean_AC_C_win = np.mean(gaps_AC_C_win) if gaps_AC_C_win else 0

    mean_AB_BC_win = np.mean(gaps_AB_BC_win) if gaps_AB_BC_win else 0
    mean_AC_BC_win = np.mean(gaps_AC_BC_win) if gaps_AC_BC_win else 0

    # Standard deviations
    std_AB_A_win = np.std(gaps_AB_A_win) if gaps_AB_A_win else 0
    std_AC_A_win = np.std(gaps_AC_A_win) if gaps_AC_A_win else 0

    std_AB_B_win = np.std(gaps_AB_B_win) if gaps_AB_B_win else 0
    std_AC_B_win = np.std(gaps_AC_B_win) if gaps_AC_B_win else 0

    std_AB_C_win = np.std(gaps_AB_C_win) if gaps_AB_C_win else 0
    std_AC_C_win = np.std(gaps_AC_C_win) if gaps_AC_C_win else 0

    std_AB_BC_win = np.std(gaps_AB_BC_win) if gaps_AB_BC_win else 0
    std_AC_BC_win = np.std(gaps_AC_BC_win) if gaps_AC_BC_win else 0

    return {
        "P(A wins)": p_A_win,
        # "P(B wins)": p_B_win,
        # "P(C wins)": p_C_win,
        "P(BC wins)": p_BC_win,
        "Mean gap g_AB": mean_g_AB,
        "Mean gap g_AC": mean_g_AC,
        "Std gap g_AB": std_g_AB,
        "Std gap g_AC": std_g_AC,
        "Mean gap g_AB | A wins": mean_AB_A_win,
        "Mean gap g_AC | A wins": mean_AC_A_win,
        "Mean gap g_AB | BC wins": mean_AB_BC_win,
        "Mean gap g_AC | BC wins": mean_AC_BC_win,
        "Std gap g_AB | A wins": std_AB_A_win,
        "Std gap g_AC | A wins": std_AC_A_win,
        "Std gap g_AB | BC wins": std_AB_BC_win,
        "Std gap g_AC | BC wins": std_AC_BC_win,
    }

def plot_sample_path(history, gap_history, pit_history):

    laps = [h[0] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # TOP: pit strategy
    # horizontal dashed lines
    # ax1.hlines(1, 1, N, linestyles='dashed')
    # ax1.hlines(0, 1, N, linestyles='dashed')
    ax1.hlines([2,1,0], 1, N, linestyles='dashed', color='black')  # horizontal guides for C, A, B


    # color mapping
    tire_colors = {
        1: "red",      # Soft
        2: "orange",   # Medium
        3: "black"     # Hard (if used)
    }

    # Plot pit stops for all 3 players
    for i, lap in enumerate(laps):
        pitA, pitB, pitC, a, b, c = pit_history[i]
        tire_A, tire_B, tire_C = history[i][1], history[i][2], history[i][3]

        # Only plot when pit happens OR first lap
        # Driver A
        if i == 0 or pitA:
            tire_label = tire_A if not pitA else a
            color = tire_colors[tire_label]
            
            ax1.scatter(lap, 2, s=120, color=color)
            ax1.text(lap, 2, 
                     "S" if tire_label == 1 else ("M" if tire_label == 2 else "H"),
                     ha='center', va='center', fontsize=9, color='white')

        # Driver B
        if i == 0 or pitB:
            tire_label = tire_B if not pitB else b
            color = tire_colors[tire_label]

            ax1.scatter(lap, 1, s=120, color=color)
            ax1.text(lap, 1, 
                     "S" if tire_label == 1 else ("M" if tire_label == 2 else "H"),
                     ha='center', va='center', fontsize=9, color='white')
            
        # Player C
        if i == 0 or pitC:
            tire_label = tire_C if not pitC else c
            color = tire_colors[tire_label]

            ax1.scatter(lap, 0, s=120, color=color)
            ax1.text(lap, 0, 
                     "S" if tire_label==1 else ("M" if tire_label==2 else "H"),
                     ha='center', va='center', fontsize=9, color='white')
            

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["C", "B", "A"])
    ax1.set_title("Race strategy")
    ax1.set_ylabel("Player")

    # BOTTOM: gaps
    g_AB = [g[0] for g in gap_history]
    g_AC = [g[1] for g in gap_history]

    ax2.plot(laps, g_AB, label="g_AB", linewidth=2)
    ax2.plot(laps, g_AC, label="g_AC", linewidth=2)
            
    ax2.axhline(0, linestyle='--', color='black')
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Time Difference [s]")
    ax2.legend()
    ax2.set_title("Time gaps relative to A")

    plt.tight_layout()
    plt.show()
