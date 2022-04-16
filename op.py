import numpy as np

#приближенное решение задачи о назначениях. Быстрый алгоритм
def get_optim_trajectory_fast(A, criterion='max'):

    def get_max_indices(svalues=np.array([]), values=np.array([]), indices=np.array([])):
        # indices[0] -бригады
        # indices[1] -скважины
        def get_max(svalues=np.array([]), values=np.array([]), indices=np.array([])):
            if indices[0].shape[0] <= 1:
                return indices[0]

            index = indices[1]
            i = 0
            n = index[0]
            if n > 0:
                size = n + 1
            else:
                size = svalues.shape[1] + 1 + n
            am = 0

            while i < size:
                x_index = svalues[indices[0], index]
                x_values = values[indices[0], x_index]

                if np.any(x_values[0] != x_values):
                    am = np.argmin(x_values)
                    return indices[0][am]
                else:
                    index = index - 1

                i += 1
                # print('i',i)
            return indices[0][am]

        arg_x = []
        arg_y = []
        row = svalues[indices[0], indices[1]]
        # print(row)
        unique = np.unique(row)
        for j in unique:
            indices_x = np.where(row == j)
            indices_y = row[indices_x]
            i_x = indices[0][indices_x]
            i_y = indices[1][indices_x]
            arg = np.argmax(values[i_x, indices_y])
            mvalue = values[i_x, indices_y][arg]
            equal = np.where(values[i_x, indices_y] == mvalue)[0]
            # print(indices_x)

            if equal.shape[0] > 1:
                index = np.array([i_x[equal], i_y[equal]])
                am = get_max(svalues, values, index)
            else:
                am = i_x[arg]

            arg_x.append(am)
            arg_y.append(j)
        return np.array([arg_x, arg_y], dtype=np.int32)

    def get_row(A=np.array([]), index=np.array([])):
        row = []
        col = []
        for j in index:
            mask = np.where(A[j, :] >= 0)
            if mask[0].shape[0] > 0:
                row.append(mask[0][-1])
                col.append(j)
        return np.array([col, row], dtype=np.int32)


    sA = A.argsort()
    trajectory_x = []
    trajectory_y = []
    s = 0
    #function = get_max_indices
    index_x = np.arange(sA.shape[0])
    index_y = np.ones(sA.shape[0]) * -1
    index = np.array([index_x, index_y], dtype=np.int32)


    while index.shape[1] > 0:
        indices = get_max_indices(sA, A, index)
        trajectory_x.extend(indices[0])
        trajectory_y.extend(indices[1])
        s_ = A[indices[0], indices[1]].sum()
        s += s_
        mask = np.isin(index[0], indices[0])
        index_x = index[0][~mask]
        mask1 = np.isin(sA, indices[1])
        sA[mask1] = -1
        index = get_row(sA, index=index_x)

        if index.shape[0] == 0:
            break
    array = np.array([trajectory_x, trajectory_y])
    sorted = array[0].argsort()
    return array[:, sorted], s

#точное решение задачи о назначениях
def get_optim_trajectory(A, criterion='max'):
    if criterion == 'max':
        A = A * -1
    indices, s_ = assignment(A)
    return indices, s_


def assignment(a=np.array([]), add=True):
    trsp = False
    if a.shape[0] > a.shape[1]:
        a = a.T
        trsp = True
    if add:
        rzeros = np.zeros(a.shape[0]).reshape(-1, 1)
        a = np.hstack((rzeros, a))
        czeros = np.zeros(a.shape[1]).reshape(1, -1)
        a = np.vstack((czeros, a))
    u = np.zeros(a.shape[0], dtype=np.float)
    v = np.zeros(a.shape[1], dtype=np.float)
    p = np.zeros(a.shape[1], dtype=np.int32)
    way = np.zeros(a.shape[1], dtype=np.int32)

    for i in np.arange(1, a.shape[0]):
        p[0] = i
        j0 = 0
        minv = np.empty(shape=a.shape[1])
        minv.fill(np.inf)
        used = np.zeros(shape=a.shape[1], dtype=bool)
        mark = True
        while mark:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in np.arange(1, a.shape[1]):
                if not used[j]:
                    cur = a[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in np.arange(a.shape[1]):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] != 0:
                mark = True
            else:
                mark = False

        mark1 = True
        while mark1:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 > 0:
                mark1 = True
            else:
                mark1 = False
    c = np.empty(a.shape[0], dtype=np.int32)
    for i in np.arange(p.shape[0]):
        c[p[i] - 1] = i - 1
    if trsp:
        return np.array([c[:-1], np.arange(a.shape[0] - 1)]), v[0]

    return np.array([np.arange(a.shape[0] - 1), c[:-1]]), v[0]

def interseption(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))

    if mask1 & mask2:
        A[0] = x
        A[1] = y
        return A
    if mask1:
        A[0] = x
        return A
    if mask2:
        A[1] = y
        return A
    if mask3:
        return A.reshape(-1, shape)
    return np.array([],dtype=float)