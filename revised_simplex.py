import numpy as np

n = 3  # number of variables
m = 3  # number of constraints


def initialize_table():
    table = np.zeros((m + 1, m + n + 1))
    return table


def add_constrants(table, constraints):
    cons = constraints.split(",")

    zeroRows = np.where(~table.any(axis=1))[0]
    if len(zeroRows) > 0:
        if "L" in cons:
            coeff_cons = list(map(int, cons[:cons.index("L")]))
            table[zeroRows[0], :cons.index("L")] = coeff_cons
            table[zeroRows[0], zeroRows[0]+n] = 1
            table[zeroRows[0], -1] = int(cons[cons.index("L")+1])
        if "G" in cons:
            coeff_cons = list(map(int, cons[:cons.index("G")]))
            coeff_cons = np.negative(coeff_cons)
            table[zeroRows[0], :cons.index("G")] = coeff_cons
            table[zeroRows[0], zeroRows[0]+n] = 1
            table[zeroRows[0], -1] = np.negative(int(cons[cons.index("G")+1]))


def add_objective(table, objective):
    obj = objective.split(",")
    obj = list(map(int, obj))

    table[-1, :n] = np.negative(obj)


def partition(table):
    A = np.reshape(table[:m, :m+n], (m, m+n))
    b = np.reshape(table[:m, -1], (m, 1))
    neg_c = np.reshape(table[-1, :m+n], (1, m+n))
    z = table[-1, -1]

    return {"A": A, "b": b, "neg_c": neg_c, "z": z}


def initialize_variable(table):
    bv = np.arange(n, n+m)
    nbv = np.arange(0, n)
    # inv_B = np.linalg.inv(B)
    # c_b = np.negative(np.take(table, bv, axis=1)[-1, :])

    variables = {"bv": bv, "nbv": nbv}

    return variables


def findEnter(partitioned, variables, P_cache):
    bv = variables["bv"]
    nbv = variables["nbv"]

    inv_B = P_cache[len(P_cache)-1]
    for i in range(len(P_cache)-2,-1,-1):
        inv_B = np.dot(inv_B,P_cache[i])

    neg_c = partitioned["neg_c"]
    A = partitioned["A"]

    c_b = np.negative(np.take(neg_c, bv, axis=1))
    c_nb = np.negative(np.take(neg_c, nbv, axis=1))
    A_nb = np.take(A, nbv, axis=1)

    next_neg_c_nb = np.dot(np.dot(c_b, inv_B), A_nb) - c_nb

    if np.amin(next_neg_c_nb) < 0:
        enter_var = nbv[np.argmin(next_neg_c_nb)]
    else:
        enter_var = -1

    return enter_var


def findLeave(partitioned, variables, enter_var, P_cache):
    inv_B = P_cache[len(P_cache)-1]
    for i in range(len(P_cache)-2,-1,-1):
        inv_B = np.dot(inv_B,P_cache[i])

    A = partitioned["A"]
    b = partitioned["b"]
    bv = variables["bv"]

    next_A_enter = np.dot(inv_B, A[:, enter_var].reshape(m, 1))
    next_b = np.dot(inv_B, b)

    ratio = next_b / next_A_enter
    valid_idx = np.where(ratio >= 0)[0]
    if len(valid_idx) == 0:
        return -1
    leave_var = bv[valid_idx[ratio[valid_idx].argmin()]]

    return leave_var


def updateBasicVariable(partitioned, variables, enter_var, leave_var, P_cache):
    inv_B = P_cache[len(P_cache)-1]
    for i in range(len(P_cache)-2,-1,-1):
        inv_B = np.dot(inv_B,P_cache[i])
    P = np.identity(m)
    bv = variables["bv"]
    nbv = variables["nbv"]
    A = partitioned["A"]
    next_A_enter = np.dot(inv_B, A[:, enter_var].reshape(m, 1))
    ind_row = np.where(bv == leave_var)[0][0]
    temp = next_A_enter[ind_row, 0]
    P[:, ind_row:ind_row+1] = np.negative(next_A_enter) / temp
    P[ind_row, enter_var] = 1/temp
    P_cache.append(P)

    bv = np.where(bv == leave_var, enter_var, bv)
    nbv = np.where(nbv == enter_var, leave_var, nbv)

    variables["bv"] = bv
    variables["nbv"] = nbv


if __name__ == "__main__":
    table = initialize_table()
    add_constrants(table, "1,2,-1,L,4")
    add_constrants(table, "1,1,1,L,6")
    add_constrants(table, "2,0,1,L,8")
    add_objective(table, "2,-1,1")

    partitioned = partition(table)
    variables = initialize_variable(table)

    P_cache = []
    P_cache.append(np.identity(m))


    enter_var = findEnter(partitioned, variables, P_cache)

    while enter_var >= 0:
        leave_var = findLeave(partitioned, variables, enter_var, P_cache)

        updateBasicVariable(partitioned, variables, enter_var, leave_var, P_cache)

        enter_var = findEnter(partitioned, variables, P_cache)

    c_b = np.negative(np.take(table, variables["bv"], axis=1)[-1, :])

    inv_B = P_cache[len(P_cache)-1]
    for i in range(len(P_cache)-2,-1,-1):
        inv_B = np.dot(inv_B,P_cache[i])

    print("z = %d" %(partitioned["z"] + np.dot(np.dot(c_b,inv_B),partitioned["b"])))
    finalA = np.take(partitioned["A"], variables["bv"], axis=1)

    res = np.linalg.solve(finalA,partitioned["b"])

    for i in range(len(variables["bv"])):
        print("x" + str(variables["bv"][i]) + " = " + str(res[i][0]))

    for i in variables["nbv"]:
        print("x" + str(i) + " = 0")