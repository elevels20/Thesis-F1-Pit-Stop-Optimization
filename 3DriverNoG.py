# NO APPENDIX G!!!
import numpy as np
from scipy.optimize import linprog
import math

# PARAMETERS
N = 5 # Number of laps 
T = [1, 2, 3] # Tire compounds. 1 = Soft, 2 = Medium, 3 = Hard
T0 = [0] + T # 0 = no pit stop

# u1 = 30 # Lifespan in number of laps for Soft tires
# u2 = 40 # Lifespan in number of laps for Medium tires
# u3 = 50 # Lifespan in number of laps for Hard tires
u1 = 3
u2 = 4
u3 = 5
u = [u1, u2, u3]

d0 = {"A":97.22, "B":97.24, "C": 97.20} # Lap time for driver A/B/C when using new soft compound tires at the beginning of the race (without pit stops, interactions, or DRS)
p0 = {"A":20.2, "B":20.0, "C": 20.4} # Additional lap time for driver A/B/C due to a pit stop

lambda_pen = 2.0
h = 0.02 # Lap time reduction between two consecutive laps attributed to fuel consumption (making the car lighter)

d_VSC = 170 # Lap time during a VSC (without pit stop)
d_SC = 200 # Lap time during a SC (without pit stop)

p_VSC = {"A":10.0, "B":10.0, "C": 10.0} # Additional lap time for driver A/B/C due to a pit stop under VSC
p_SC  = {"A":10.0, "B":10.0, "C": 10.0} #additional lap time for driver A/B/C due to a pit stop under SC

g_AB1 = -0.4 # Initial time gap between A and B
g_AC1 = -0.08 # Initial time gap between A and C

# gap discretization
# g_min = -35.0
# g_max = 35.0
# g_step = 0.04
# g_values = np.arange(g_min, g_max + g_step, g_step)

g_min = -4
g_max = 4
g_step = 2.0      
g_values = np.arange(g_min, g_max + g_step, g_step)

# Stochastic yellow flag 
l_VSC = 2 # Laps that last a VSC
l_SC = 3 # Laps that last a SC

r_VSC = 0.3 # Probability of having at least one VSC during the race
r_SC = 0.3 # Probability of having at least one SC during the race

# r = 1 - (1 - q) ** N
# q = 1 - (1 - r) ** (1/N)
q_VSC = 1 - (1 - r_VSC) ** (1/N)
q_SC = 1 - (1 - r_SC) ** (1/N)

k_VSC = 2 # Number of laps needed to be raced after the end of a VSC to enable the DRS
k_SC = 2 # Number of laps needed to be raced after the end of a SC to enable the DRS

# RANDOM VARIABLES
# Z1_vals = [0.2, 0.4, 0.6] 
# Z2_vals = [0.5, 0.7, 0.9]
# Z_prob = [1/3, 1/3, 1/3]

# TDRS_vals = [0.1, 0.3, 0.5]
# TDRS_prob = [0.2, 0.7, 0.1]

Z1_vals = [0.4]
Z2_vals = [0.7]
Z_prob = [1.0]

TDRS_vals = [0.3]
TDRS_prob = [1.0]

# state = (tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC, y_VSC, y_SC, y_DRS)

# FUNCTIONS

# Tire-wear function
def tire_wear(tire, w):
    if tire == 1:
        return 0.004 * (w - 1) + 0.005 * ((w - 1) ** 2)
    elif tire == 2:
        return 0.4 + 0.20 * w + 0.008 * (w ** 2)
    elif tire == 3:
        return 0.9 + 0.10 * w + 0.0001 * (w ** 2)

# Interaction function
def interaction_A(eps_AB, eps_AC, Z1, Z2):
    interaction_AB = 0
    interaction_AC = 0

    if eps_AB < 0:
        interaction_AB = np.exp(-lambda_pen * abs(eps_AB)) * Z1
    else:
        interaction_AB = np.exp(-lambda_pen * abs(eps_AB)) * Z2

    if eps_AC < 0:
        interaction_AC = np.exp(-lambda_pen * abs(eps_AC)) * Z1
    else:
        interaction_AC = np.exp(-lambda_pen * abs(eps_AC)) * Z2

    return interaction_AB + interaction_AC

def interaction_B(eps_AB, eps_BC, Z1, Z2):
    interaction_BA = 0
    interaction_BC = 0

    if eps_AB >= 0:
        interaction_BA = np.exp(-lambda_pen * abs(eps_AB)) * Z1
    else:
        interaction_BA = np.exp(-lambda_pen * abs(eps_AB)) * Z2

    if eps_BC < 0:
        interaction_BC = np.exp(-lambda_pen * abs(eps_BC)) * Z1
    else:
        interaction_BC = np.exp(-lambda_pen * abs(eps_BC)) * Z2

    return interaction_BA + interaction_BC

def interaction_C(eps_AC, eps_BC, Z1, Z2):
    interaction_CA = 0
    interaction_CB = 0

    if eps_AC >= 0:
        interaction_CA = np.exp(-lambda_pen * abs(eps_AC)) * Z1
    else:
        interaction_CA = np.exp(-lambda_pen * abs(eps_AC)) * Z2

    if eps_BC >= 0:
        interaction_CB = np.exp(-lambda_pen * abs(eps_BC)) * Z1
    else:
        interaction_CB = np.exp(-lambda_pen * abs(eps_BC)) * Z2

    return interaction_CA + interaction_CB

# Lap time functions
def lap_time_VSC(driver, pit):    
    return d_VSC + (p_VSC[driver] if pit else 0)

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

def lap_time_SC(g_AB, g_AC, g_BC, driver, pitA, pitB, pitC):
    order = driver_order(g_AB, g_AC, g_BC, pitA, pitB, pitC)

    if driver == "A":
        IA = 1 if pitA else 0
        if driver == order[0]: # A is P1
            lap_time = d_SC + p_SC[driver] * IA
        else: # A is P2 or P3
            g = g_AB if order[order.index(driver) - 1] == "B" else g_AC # A is immediately behind ...
            lap_time = d_SC + p_SC[driver] * IA - g + 0.5        
    elif driver == "B":
        IB = 1 if pitB else 0
        if driver == order[0]: # B is P1
            lap_time = d_SC + p_SC[driver] * IB
        else: # b is P2 or P3
            g = g_AB if order[order.index(driver) - 1] == "A" else -g_BC
            lap_time = d_SC + p_SC[driver] * IB + g + 0.5
    else: # driver == "C"
        IC = 1 if pitC else 0
        if driver == order[0]: # C is P1
            lap_time = d_SC + p_SC[driver] * IC
        else: # C is P2 or P3
            g = g_AC if order[order.index(driver) - 1] == "A" else g_BC
            lap_time = d_SC + p_SC[driver] * IC + g + 0.5

    return lap_time

def lap_time_no_yellow_flag(driver, n, tire_n, w, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS): 
    IA = 1 if pitA else 0
    IB = 1 if pitB else 0
    IC = 1 if pitC else 0

    drs_AB = 1 if (0 <= g_AB <= 1) and y_DRS == 0 else 0
    drs_BA = 1 if (-1 <= g_AB <= 0) and y_DRS == 0 else 0
    drs_AC = 1 if (0 <= g_AC <= 1) and y_DRS == 0 else 0
    drs_CA = 1 if (-1 <= g_AC <= 0) and y_DRS == 0 else 0
    drs_BC = 1 if (0 <= g_BC <= 1) and y_DRS == 0 else 0
    drs_CB = 1 if (-1 <= g_AB <= 0) and y_DRS == 0 else 0

    eps_AB = g_AB + p0["A"] * IA - T_DRS * drs_AB - (p0["B"] * IB - T_DRS * drs_BA)
    eps_AC = g_AC + p0["A"] * IA - T_DRS * drs_AC - (p0["C"] * IC - T_DRS * drs_CA)
    eps_BC = g_BC + p0["B"] * IB - T_DRS * drs_BC - (p0["C"] * IC - T_DRS * drs_CB)

    if driver == "A":
        eta = interaction_A(eps_AB, eps_AC, Z1, Z2)
        pit = pitA
        drs = 1 if ((0 <= g_AB <= 1 or 0 <= g_AC <= 1) and y_DRS == 0) else 0
    elif driver == "B":
        eta = interaction_B(eps_AB, eps_BC, Z1,Z2)
        pit = pitB
        drs = 1 if ((-1 <= g_AB <= 0 or 0 <= g_BC <= 1) and y_DRS == 0) else 0
    else:
        eta = interaction_C(eps_AC, eps_BC, Z1,Z2)
        pit = pitC
        drs = 1 if ((-1 <= g_AC <= 0 or -1 <= g_BC <= 0) and y_DRS == 0) else 0

    gamma = d0[driver] + (p0[driver] if pit else 0) + tire_wear(tire_n, w) - h * n
    
    return gamma + eta - T_DRS * drs

def final_lap_time(y_VSC, y_SC, driver, n, tire_n, w, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS):
    if y_VSC + y_SC == 0:
        return lap_time_no_yellow_flag(driver, n, tire_n, w, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS)
    elif y_VSC > 0:
        if driver == "A":
            pit = pitA
        elif driver == "B":
            pit = pitB
        else: 
            pit = pitC
        return lap_time_VSC(driver, pit)
    elif y_SC > 0:
        return lap_time_SC(g_AB, g_AC, g_BC, driver, pitA, pitB, pitC)
    
# State functions
def t_next(tire, decision):
    if decision == 0:
        return tire
    else:
        return decision
    
def w_next(w, decision, Y_VSC, Y_SC, u):
    I_driver = 1 if decision == 0 else 0
    I_Y = 1 if Y_VSC + Y_SC == 0 else 0

    return min(w * I_driver + I_Y, u + 1)

def m_next(m, decision, tire):
    if decision != 0 and decision != tire:
        return max(m, 1)
    else:
        return max(m, 0)
    
def g_AB_next(y_VSC, y_SC, n, tire_A_n, tire_B_n, wA, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS):
    return g_AB + final_lap_time(y_VSC, y_SC, "A", n, tire_A_n, wA, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS) - final_lap_time(y_VSC, y_SC, "B", n, tire_B_n, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS)
    
def g_AC_next(y_VSC, y_SC, n, tire_A_n, tire_C_n, wA, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS):
    return g_AC + final_lap_time(y_VSC, y_SC, "A", n, tire_A_n, wA, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS) - final_lap_time(y_VSC, y_SC, "C", n, tire_C_n, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, Z1, Z2, T_DRS)
    
def g_BC_next(g_AB_n, g_AC_n):
    return g_AC_n - g_AB_n
    
def discretize_gap(g):
    return min(g_values, key=lambda x: abs(x - g))

def y_VSC_next(Y_VSC):
    return Y_VSC

def y_SC_next(Y_SC):
    return Y_SC

def y_DRS_next(y_DRS, Y_VSC, Y_SC):
    I_Y = 1 if Y_VSC + Y_SC == 0 else 0
    I_Y_VSC = 1 if Y_VSC > 0 else 0
    I_Y_SC = 1 if Y_SC > 0 else 0

    return max(y_DRS - 1, 0) * I_Y + (Y_VSC + k_VSC) * I_Y_VSC + (Y_SC + k_SC) * I_Y_SC

def yellow_transitions(y_VSC, y_SC):
    if y_VSC > 0:
        return [((y_VSC - 1, 0), 1.0)]
    if y_SC > 0:
        return [((0, y_SC - 1), 1.0)]

    return [
        ((l_VSC, 0), q_VSC),
        ((0, l_SC), q_SC),
        ((0, 0), 1 - q_VSC - q_SC)
    ]
def state_next(tire_A, wA, mA, tire_B, wB, mB, tire_C, wC, mC, g_AB, g_AC, y_VSC, y_SC, y_VSC_n, y_SC_n, y_DRS, n, decisionA, decisionB, decisionC, z1, z2, t_DRS):
    tire_A_n = t_next(tire_A, decisionA)
    tire_B_n = t_next(tire_B, decisionB)
    tire_C_n = t_next(tire_C, decisionC)

    wA_n = w_next(wA, decisionA, y_VSC, y_SC, u[tire_A - 1])
    wB_n = w_next(wB, decisionB, y_VSC, y_SC, u[tire_B - 1])
    wC_n = w_next(wC, decisionC, y_VSC, y_SC, u[tire_C - 1])

    mA_n = m_next(mA, decisionA, tire_A)
    mB_n = m_next(mB, decisionB, tire_B)
    mC_n = m_next(mC, decisionC, tire_C)

    pitA = True if decisionA != 0 else False
    pitB = True if decisionB != 0 else False
    pitC = True if decisionC != 0 else False

    g_BC = g_AC - g_AB

    g_AB_n = g_AB_next(y_VSC, y_SC, n, tire_A_n, tire_B_n, wA, wB, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, z1, z2, t_DRS)
    g_AC_n = g_AC_next(y_VSC, y_SC, n, tire_A_n, tire_C_n, wA, wC, pitA, pitB, pitC, g_AB, g_AC, g_BC, y_DRS, z1, z2, t_DRS)
    g_BC_n = g_BC_next(g_AB_n, g_AC_n)
    
    g_AB_n = discretize_gap(g_AB_n)
    g_AC_n = discretize_gap(g_AC_n)
    g_BC_n = discretize_gap(g_BC_n)

    y_DRS_n = y_DRS_next(y_DRS, y_VSC, y_SC)

    return (tire_A_n, wA_n, mA_n, tire_B_n, wB_n, mB_n, tire_C_n, wC_n, mC_n, g_AB_n, g_AC_n, y_VSC_n, y_SC_n, y_DRS_n)

# Stochastic dynamic programming
def H(w, u, m): 
    """
    Returns whether the driver finishes the race adequately without using tires 
    longer than their lifespans and using at least two different tire compounds.
    """
    return w <= u and m == 1

def V_end(tire_A, wA, mA, tire_B, wB, mB, g):
    Ha = H(wA, u[tire_A - 1], mA)
    Hb = H(wB, u[tire_B - 1], mB)

    if Ha:
        if Hb:
            return g
        else:
            return - math.inf
    elif Hb:
        return math.inf
    else:
        return 0

# state = (tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_DRS)

def generate_states(n):
    states = []
    max_y_DRS = max(k_VSC + l_VSC, k_SC + l_SC)

    for tire_A in T:
        for tire_B in T:

            # for wA in range(1, u[tire_A - 1] + 2):
            for wA in range(u[tire_A - 1] + 2):
                if wA > n: # tire wear cannot be higher than current lap
                    continue
                # for wB in range(1, u[tire_B - 1] + 2):
                for wB in range(u[tire_B - 1] + 2):
                    if wB > n: # tire wear cannot be higher than current lap
                        continue

                    for mA in range(2):
                        if mA == 1 and n == 1: # mA = 1 only if pit happened in a previous lap, this cannot hapen in the first lap
                            continue
                        for mB in range(2):
                            if mB == 1 and n == 1: # mB = 1 only if pit happened in a previous lap, this cannot hapen in the first lap
                                continue

                            for g in g_values:

                                for y_VSC in range(l_VSC + 1):
                                    for y_SC in range(l_SC + 1):

                                        # Eliminate invalid combos
                                        if y_VSC > 0 and y_SC > 0:
                                            continue

                                        for y_DRS in range(max_y_DRS + 1):

                                            states.append((
                                                tire_A, wA, mA,
                                                tire_B, wB, mB,
                                                g,
                                                y_VSC, y_SC,
                                                y_DRS
                                            ))

    return states

# Stochastic Dynamic Programming Algorithm
def solve_SDP():
    # Step 3: Compute 𝑉 ′^{*} _N+1(s_n) for all s_n ∈ S_N+1
    states_final = generate_states(N + 1) 
    V = {state: 0 for state in states_final}
    for state in states_final:
        tire_A, wA, mA, tire_B, wB, mB, g, _, _, _ = state
        V[state] = V_end(tire_A, wA, mA, tire_B, wB, mB, g)

    # Policies
    xA_star = {}
    xB_star = {}

    RV_combinations = []

    for t_DRS, p_DRS in zip(TDRS_vals, TDRS_prob):
        for z1, p_z1 in zip(Z1_vals, Z_prob):
            for z2, p_z2 in zip(Z2_vals, Z_prob):
                prob = p_DRS * p_z1 * p_z2
                RV_combinations.append((z1, z2, t_DRS, prob))

    # state_next_cache = {}

    for n in range(N, 0, -1):
        V_new = {}
        states = generate_states(n)
        for state in states:
            tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_DRS = state

            # Step 6: compute V'_n(s,a,b) for all a, b ∈ T0.
            V_prime = {}
            for a in T0:
                for b in T0:
                    val = 0

                    for (z1, z2, t_DRS, p_rv) in RV_combinations:
                        for (y_VSC_n, y_SC_n), p_y in yellow_transitions(y_VSC, y_SC):
                            # key = (tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_VSC_n, y_SC_n, y_DRS, n, a, b, z1, z2, t_DRS)
                            state_n = state_next(tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_VSC_n, y_SC_n, y_DRS, n, a, b, z1, z2, t_DRS)

                            # if key not in state_next_cache:
                                # state_next_cache[key] = state_next(tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_VSC_n, y_SC_n, y_DRS, n, a, b, z1, z2, t_DRS)
                            
                            # state_n = state_next_cache[key]

                            prob = p_rv * p_y
                            val += prob * V[state_n]                                    

                    V_prime[(a, b)] = val

            if g < 0: # A is leader (minimize g)
                # Step 8: compute x_n^{B*}(s,a) for all a ∈ T0.
                b_star = {a: max(T0, key=lambda b: V_prime[(a, b)]) for a in T0}

                # Step 9: compute x_n^{A*}(s)
                a_star = min(T0, key=lambda a: V_prime[(a, b_star[a])])

                # Step 10: value update
                V_new[state] = V_prime[(a_star, b_star[a_star])]

                xA_star[(n, state)] = a_star
                xB_star[(n, state)] = b_star[a_star]
            
            else: # B is leader (maximize g)
                # Step 12: compute x_n^{A*}(s,b) for all b ∈ T0.
                a_star = {b: min(T0, key=lambda a: V_prime[(a, b)]) for b in T0}

                # Step 13: compute x_n^{B*}(s)
                b_star = max(T0, key=lambda b: V_prime[(a_star[b], b)])

                # Step 14: value update
                V_new[state] = V_prime[(a_star[b_star], b_star)]

                xA_star[(n, state)] = a_star[b_star]
                xB_star[(n, state)] = b_star
    
        V = V_new
    # return V, xA_star, xB_star

    T_list = list(T)
    nT = len(T_list)

    # Step 15: build payoff matrix U'
    U = np.zeros((nT, nT))

    g_init = discretize_gap(g1)

    for i, tA in enumerate(T_list):
        for j, tB in enumerate(T_list):
            s1 = (tA, 0, 0, tB, 0, 0, g_init, 0, 0, 2)
            U[i, j] = V[s1]

    # Step 16: solve zero-sum game via LP   
    # Player B (E.3)

    # Variables: [π_B (nT), ρ]
    c = np.zeros(nT + 1) # Coefficients of the objective function to minimize.
    c[-1] = -1  # maximize ρ

    A_ub = [] # Coefficients for the inequality constraints 
    b_ub = [] # Right-hand side values for the inequality constraints.

    # ρ - (U π_B)_i ≤ 0  →  -U[i,:] π_B + ρ ≤ 0
    for i in range(nT):
        A_ub.append([-U[i, j] for j in range(nT)] + [1])
        b_ub.append(0)

    A_eq = [[1]*nT + [0]] # Coefficients for the equality constraints (Ax = b).
    b_eq = [1] # Right-hand side value for the equality constraints.

    bounds = [(0,1)]*nT + [(None, None)] # Specifies the bounds for each variable in the form of a sequence of tuples.

    resB = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    pi_B = resB.x[:-1]

    # Player A (E.4)
    # Variables: [π_A (nT), ρ]
    c = np.zeros(nT + 1)
    c[-1] = 1  # minimize ρ

    A_ub = []
    b_ub = []

    # ρ - (U^T π_A)_i ≥ 0  →  (U^T π_A)_i - ρ ≤ 0
    for i in range(nT):
        A_ub.append([U[j, i] for j in range(nT)] + [-1])
        b_ub.append(0)

    A_eq = [[1]*nT + [0]]
    b_eq = [1]

    bounds = [(0,1)]*nT + [(None, None)]

    resA = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    pi_A = resA.x[:-1]

    return U, pi_A, pi_B, xA_star, xB_star

# Run
# U, pi_A, pi_B = solve_SDP()

# print(f"Payoff matrix U': \n{U}")
# print(f"π^A (mixed strategy): \n{pi_A}")
# print(f"π^B (mixed strategy):\n{pi_B}")

# if __name__ == '__main__':
    # V, xA, xB = solve_algorithm1()
    # print('Solved Algorithm 1')
    # print('Number of states:', len(V))

def simulate_race(pi_A, pi_B, xA_star, xB_star):
    # Initial state
    tA = np.random.choice(T, p=pi_A)
    tB = np.random.choice(T, p=pi_B)

    g = discretize_gap(g1)

    state = (tA, 0, 0, tB, 0, 0, g, 0, 0, 2)

    history = []
    gap_history = []
    yellow_history = []
    pit_history = []

    # Simulate laps
    for n in range(1, N + 1):

        tire_A, wA, mA, tire_B, wB, mB, g, y_VSC, y_SC, y_DRS = state

        # Optimal decisions from SDP
        a = xA_star[(n, state)]
        b = xB_star[(n, state)]

        pitA = (a != 0)
        pitB = (b != 0)

        history.append((n, tire_A, tire_B))
        gap_history.append(g)
        yellow_history.append((y_VSC, y_SC))
        pit_history.append((pitA, pitB, a, b))

        # Sample stochastic variables
        z1 = np.random.choice(Z1_vals, p=Z_prob)
        z2 = np.random.choice(Z2_vals, p=Z_prob)
        t_DRS = np.random.choice(TDRS_vals, p=TDRS_prob)

        # Sample yellow flag transition
        transitions = yellow_transitions(y_VSC, y_SC)
        probs = [p for (_, p) in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        (y_VSC_n, y_SC_n), _ = transitions[idx]

        # Compute next state
        state = state_next(
            tire_A, wA, mA,
            tire_B, wB, mB,
            g,
            y_VSC, y_SC,
            y_VSC_n, y_SC_n,
            y_DRS,
            n,
            a, b,
            z1, z2, t_DRS
        )

    # Final outcome
    final_gap = state[6]

    if final_gap < 0:
        winner = "A"
    elif final_gap > 0:
        winner = "B"
    else:
        winner = "tie"

    return final_gap, winner, history, gap_history, yellow_history, pit_history

def run_simulations(U, pi_A, pi_B, xA_star, xB_star, n_sim=10000):
    results = []

    for _ in range(n_sim):
        g_final, winner, _, _, _, _ = simulate_race(pi_A, pi_B, xA_star, xB_star)
        results.append((g_final, winner))

    gaps = np.array([r[0] for r in results])
    winners = [r[1] for r in results]

    # Probabilities
    p_A_win = np.mean([w == "A" for w in winners])
    p_B_win = np.mean([w == "B" for w in winners])
    p_tie = np.mean([w == "tie" for w in winners])

    # Unconditional stats
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    # Conditional stats
    gaps_A_win = gaps[gaps < 0]
    gaps_B_win = gaps[gaps > 0]

    mean_A_win = np.mean(gaps_A_win) if len(gaps_A_win) > 0 else 0
    mean_B_win = np.mean(gaps_B_win) if len(gaps_B_win) > 0 else 0

    std_A_win = np.std(gaps_A_win) if len(gaps_A_win) > 0 else 0
    std_B_win = np.std(gaps_B_win) if len(gaps_B_win) > 0 else 0

    return {
        "P(A wins)": p_A_win,
        "P(B wins)": p_B_win,
        "P(tie)": p_tie,
        "Mean gap": mean_gap,
        "Std gap": std_gap,
        "Mean gap | A wins": mean_A_win,
        "Mean gap | B wins": mean_B_win,
        "Std gap | A wins": std_A_win,
        "Std gap | B wins": std_B_win,
    }

def plot_sample_path(history, gap_history, yellow_history, pit_history):

    laps = [h[0] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # TOP: pit strategy
    # horizontal dashed lines
    ax1.hlines(1, 1, N, linestyles='dashed')
    ax1.hlines(0, 1, N, linestyles='dashed')

    # color mapping
    tire_colors = {
        1: "red",      # Soft
        2: "orange",   # Medium
        3: "black"     # Hard (if used)
    }

    for i, lap in enumerate(laps):
        pitA, pitB, a, b = pit_history[i]

        # Only plot when pit happens OR first lap
        # Driver A
        if i == 0 or pitA:
            tire_label = history[i][1] if not pitA else a
            color = tire_colors[tire_label]
            
            ax1.scatter(lap, 1, s=120, color=color)
            ax1.text(lap, 1, 
                     "S" if tire_label == 1 else ("M" if tire_label == 2 else "H"),
                     ha='center', va='center', fontsize=9, color='white')

        # Driver B
        if i == 0 or pitB:
            tire_label = history[i][2] if not pitB else b
            color = tire_colors[tire_label]

            ax1.scatter(lap, 0, s=120, color=color)
            ax1.text(lap, 0, 
                     "S" if tire_label == 1 else ("M" if tire_label == 2 else "H"),
                     ha='center', va='center', fontsize=9, color='white')

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["B", "A"])
    ax1.set_title("Race strategy")

    # BOTTOM: gap
    ax2.plot(laps, gap_history, linewidth=2)

     # Merge consecutive yellow flags into one block
    start_idx = None
    current_flag = 0  # 0 = none, 1 = VSC, 2 = SC

    for i, (yV, yS) in enumerate(yellow_history + [(0, 0)]):  # append dummy to flush last block
        flag = 1 if yV > 0 else (2 if yS > 0 else 0)

        if flag != current_flag:
            if current_flag != 0:
                # Plot the previous block
                color = 'yellow' if current_flag == 1 else 'orange'
                ax2.axvspan(laps[start_idx], laps[i-1]+1, alpha=0.3, color=color)
                # Label in the middle of the block
                ax2.text((laps[start_idx] + laps[i-1]+1)/2, max(gap_history)*0.95, 
                         "VSC" if current_flag == 1 else "SC",
                         ha='center', va='top', fontsize=9, color='black', rotation=90)
            # start new block
            start_idx = i
            current_flag = flag
            
    ax2.axhline(0, linestyle='--')
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Time Difference [s]")

    plt.tight_layout()
    plt.show()
    
def get_sample_no_yellow(pi_A, pi_B, xA_star, xB_star):
    while True:
        _, _, h, g, y, p = simulate_race(pi_A, pi_B, xA_star, xB_star)
        if all(yV == 0 and yS == 0 for (yV, yS) in y):
            return h, g, y, p

# def get_sample_with_VSC(pi_A, pi_B, xA_star, xB_star):
    # while True:
        # _, _, h, g, y, p = simulate_race(pi_A, pi_B, xA_star, xB_star)
        # if any(yV > 0 for (yV, yS) in y):
            # return h, g, y, p
        
def get_sample_with_yellow(pi_A, pi_B, xA_star, xB_star):
    while True:
        _, _, h, g, y, p = simulate_race(pi_A, pi_B, xA_star, xB_star)
        # Check for any yellow flag (VSC or SC)
        if any(yV > 0 or yS > 0 for (yV, yS) in y):
            return h, g, y, p