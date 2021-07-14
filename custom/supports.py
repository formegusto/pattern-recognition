import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

outlier_pca = PCA(n_components=1)


def remove_outlier_single(datas, season_idx):
    copy_datas = datas[season_idx][datas[season_idx].columns.difference(
        ['year', 'month', 'day'])].copy()
    copy_datas.set_index('date', inplace=True)
    pca_datas = pd.DataFrame(
        outlier_pca.fit_transform(copy_datas), columns=['y'])
    pca_datas.index = copy_datas.index

    Q1 = np.percentile(pca_datas['y'], 25)
    Q3 = np.percentile(pca_datas['y'], 75)
    IQR = Q3 - Q1

    outlier_step = 1.5 * IQR
    outlier_step

    remove_idx = pca_datas[(pca_datas['y'] < Q1 - IQR)
                           | (pca_datas['y'] > Q3 + IQR)].index

    return remove_idx.values


def remove_outlier_multi(datas, season_idx):
    copy_datas = datas[season_idx][datas[season_idx].columns.difference(
        ['year', 'month', 'day'])].copy()
    copy_datas.set_index('date', inplace=True)

    remove_idxes = []
    while True:
        pca_datas = pd.DataFrame(outlier_pca.fit_transform(
            copy_datas[~copy_datas.index.isin(remove_idxes)]), columns=['y'])
        pca_datas.index = copy_datas[~copy_datas.index.isin(
            remove_idxes)].index

        Q1 = np.percentile(pca_datas['y'], 25)
        Q3 = np.percentile(pca_datas['y'], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR
        outlier_step

        remove_idx = pca_datas[(pca_datas['y'] < Q1 - IQR)
                               | (pca_datas['y'] > Q3 + IQR)].index
        if len(remove_idx) == 0:
            break
        else:
            remove_idxes.extend(remove_idx)

    return remove_idxes


def elbow_k_check(check_size, points):
    inertia_arr = []
    diff_dict = {}
    k_range = range(2, check_size)

    ''' elbow Check '''
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit_transform(points)

        inertia = kmeans.inertia_
    #    print('k :', k, 'interia :', interia)

        inertia_arr.append(inertia)
        if k > 2:
            diff_dict[k] = inertia_arr[k - 3] - inertia_arr[k - 2]

    inertia_arr = np.array(inertia_arr)
    K = max(diff_dict, key=diff_dict.get)

    return (K, k_range, inertia_arr)
