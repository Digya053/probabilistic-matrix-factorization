import sys
import numpy as np


def PMF(data, lm=2, sigma=0.1, d=5, iterations=50):
    # dimension of matrix M
    Nu = int(np.amax(data[:, 0]))
    Nv = int(np.amax(data[:, 1]))

    # i_ui = index set of objects rated by user i
    # i_vj = index set of users who rated object j
    i_ui = [data[data[:, 0] == i + 1]
            [:, 1].astype(np.int64) for i in range(Nu)]
    i_vj = [data[data[:, 1] == j + 1]
            [:, 0].astype(np.int64) for j in range(Nv)]

    U, V, M = get_matrices(data, lm, d, Nu, Nv)
    U, U_matrices = update_U(iterations, lm, sigma, d, Nu, i_ui, U, V, M)
    V, V_matrices = update_V(iterations, lm, sigma, d, Nv, i_vj, U, V, M)
    L = map_objective(data, iterations, lm, sigma, U, V)

    return L, U_matrices, V_matrices


def get_matrices(data, lm, d, Nu, Nv):
    # matrix U = Nu * dim, V = Nv * dim
    U = np.zeros((Nu, d))
    V = np.random.normal(0, np.sqrt(1 / lm), (Nv, d))
    M = np.zeros((Nu, Nv))

    for val in data:
        M[int(val[0]) - 1, int(val[1]) - 1] = val[2]

    return U, V, M


def update_U(iterations, lm, sigma, d, Nu, i_ui, U, V, M):
    U_matrices = []

    for iteration in range(iterations):
        for i in range(Nu):
            Vj = V[i_ui[i] - 1]
            temp1 = lm * sigma * np.eye(d) + np.dot(Vj.T, Vj)
            temp2 = (Vj * M[i, i_ui[i] - 1][:, None]).sum(axis=0)
            U[i] = np.dot(np.linalg.inv(temp1), temp2)
        U_matrices.append(U)

    return U, U_matrices


def update_V(iterations, lm, sigma, d, Nv, i_vj, U, V, M):
    V_matrices = []

    for iteration in range(iterations):
        for j in range(Nv):
            Ui = U[i_vj[j] - 1]
            temp1 = lm * sigma * np.eye(d) + np.dot(Ui.T, Ui)
            temp2 = (Ui * M[i_vj[j] - 1, j][:, None]).sum(axis=0)
            V[j] = np.dot(np.linalg.inv(temp1), temp2)
        V_matrices.append(V)

    return V, V_matrices


def map_objective(data, iterations, lm, sigma, U, V):
    L = []
    objective = np.zeros((iterations, 1))

    t2 = lm * 0.5 * (((np.linalg.norm(U, axis=1))**2).sum())
    t3 = lm * 0.5 * (((np.linalg.norm(V, axis=1))**2).sum())

    for iteration in range(iterations):
        t1 = 0
        for val in data:
            i = int(val[0])
            j = int(val[1])
            t1 = t1 + (val[2] - np.dot(U[i - 1, :], V[j - 1, :]))**2
        t1 = t1 / (2 * sigma)
        l = -t1 - t2 - t3
        L.append(l)

    return L


def main():
    train_data = np.genfromtxt(sys.argv[1], delimiter=",")

    L, U_matrices, V_matrices = PMF(train_data)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


if __name__ == "__main__":
    main()
