import operator
from scipy.spatial import distance
import random

import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B)/(norm(A) * norm(B))


def euclidean(A, B):
    distance = 0
    for idx, value in enumerate(A):
        distance += ((B[idx] - A[idx]) ** 2)
    distance = np.sqrt(distance)
    return distance


def min_max_normalization(list):
    return [
        (val - list.min()) /
        (list.max() - list.min())
        for val in
        list.values
    ]


class PatternKMeans:
    def __init__(self, init_datas):
        self.datas = init_datas
        self.mean_pattern = self.datas.T.mean()
        self.labels = []
        self.cluster_dict = {}
        self.cluster_info = pd.DataFrame()
        self.visual_datas = pd.DataFrame()
        self.cluster_cost_map = {}

    def set_K(self, K):
        print("K is {}".format(K))
        self.K = K

    # X (distance), Y (cosine)
    def dimension_reduction(self):
        dr_datas = pd.DataFrame(columns=['x', 'y'])

        for idx in self.datas.copy():
            dr_datas.loc[idx] = [
                distance.euclidean(self.datas[idx].values,
                                   self.mean_pattern
                                   ),
                cos_sim(self.datas[idx].values, self.mean_pattern)
            ]

        dr_datas['x'] = min_max_normalization(dr_datas['x'])
        dr_datas['y'] = min_max_normalization(dr_datas['y'])

        return dr_datas

    # Remove Outlier
    def remove_outlier(self):
        datas = self.dimension_reduction()

        outlier_range = 1.5

        # sim_info scaler (Min-Max Normalization)

        datas['x'] = min_max_normalization(datas['x'])
        datas['y'] = min_max_normalization(datas['y'])
        # print(datas['y'].values)

        dis_info = {
            "Q1": np.percentile(datas['x'], 25),
            "Q3": np.percentile(datas['x'], 75),
        }
        dis_info["IQR"] = dis_info["Q3"] - dis_info["Q1"]
        dis_info["step"] = outlier_range * dis_info["IQR"]
        dis_info["check"] = dis_info["Q3"] + dis_info["step"]
        print("dis_info", dis_info)

        sim_info = {
            "Q1": np.percentile(datas['y'], 25),
            "Q3": np.percentile(datas['y'], 75)
        }
        sim_info["IQR"] = sim_info["Q3"] - sim_info["Q1"]
        sim_info["step"] = outlier_range * sim_info["IQR"]
        sim_info["check"] = sim_info["Q1"] - sim_info["step"]
        print("sim_info", sim_info)

        remove_outlier_index = datas[
            ((datas['x'] > dis_info['check'])
             |
             (datas['y'] < sim_info['check']))
        ].copy().index

        # 1자 멤버 추가적으로 제거
        print([col if len(set(self.datas[col].values))
              == 1 else None for col in self.datas])

        og_length = len(self.datas.columns)

        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_outlier_index)]
        new_datas = new_datas.T
        self.datas = new_datas.copy()

        print("""-------Remove Outlier Success-------\n  Target length {} -> {}""".format(og_length,
              len(self.datas.columns)))

        return new_datas, datas

    # Remove Outlier
    def remove_cluster_outlier(self, idxes):
        remove_outliers = []
        for num in idxes:
            remove_outliers.extend(self.cluster_info[
                self.cluster_info['label'] == num
            ].index)

        og_length = len(self.datas.columns)
        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_outliers)]
        new_datas = new_datas.T
        self.datas = new_datas.copy()

        print("""-------Remove Outlier Success-------\n  Target length {} -> {}""".format(og_length,
              len(self.datas.columns)))
    # cluser has one member

    def has_one_member(self):
        cluster_info = self.cluster_info.copy()

        _count = cluster_info['label'].groupby(
            cluster_info['label']).count()

        Q1 = np.percentile(_count, 25)
        Q3 = np.percentile(_count, 75)
        IQR = Q3 - Q1
        step = 1.5 * IQR

        print(_count)
        print("Q1: {}, Q3: {}, IQR: {}, step:{}".format(Q1, Q3, IQR, step))
        check = np.where(cluster_info['label'].groupby(
            cluster_info['label']).count().values <= (1))
        isIn = len(check[0]) >= 1

        if isIn:
            rtn_index = check[0]
        else:
            rtn_index = []
        return isIn, rtn_index

    # ECV = True, Loop until ECV >= 90
    # But Prev == Now is End
    # init is "random, worst, best"

    def run(self, init_condition="worst", ECV=False, remove_outlier=False):
        print("K Is {}".format(self.K))
        divide_cluster = False

        if remove_outlier:
            self.remove_outlier()

        while True:
            sequence = 0
            row_datas = self.datas.T.copy()
            print("클러스터 분리 여부 : {}".format(divide_cluster))
            if not divide_cluster:
                self.cluster_dict = {}
            prev_ecv = 0
            while True:
                print("---Now {}---".format(sequence))
                if not divide_cluster:
                    self.cluster_cost_map = {}

                # 첫 클러스터링
                # Process , 첫 K 초기화
                if sequence == 0:
                    cluster_index = []

                    if not divide_cluster:
                        print("First K Init")
                        if init_condition == "random":
                            init_k = random.randint(
                                0, len(row_datas.index) - 1)
                            init_cluster_index = row_datas.index[init_k]
                        elif init_condition == "best":
                            dr_datas = self.dimension_reduction()
                            init_cluster_index = dr_datas.sort_values(
                                ['x', 'y'], ascending=[True, False]).index[0]
                        else:
                            dr_datas = self.dimension_reduction()
                            init_cluster_index = dr_datas.sort_values(
                                ['x', 'y'], ascending=[False, True]).index[0]

                        init_cluster = row_datas.loc[init_cluster_index].copy()
                        cluster_index.append(init_cluster_index)
                        self.cluster_dict[0] = init_cluster
                        self.cluster_cost_map[0] = 0
                    print("나머지")
                    # 나머지 K 초기화
                    sim_arr = []
                    sim_sort_arr = []
                    for k in range(len(self.cluster_dict.keys()), self.K):
                        sim_arr = ['dis_{}'.format(idx) for idx in range(0, k)]
                        sim_arr.extend(['cos_{}'.format(idx)
                                        for idx in range(0, k)])
                        sim_sort_arr = [
                            True if (len(sim_arr) / 2) <= idx else False
                            for idx in range(0, k * 2)
                        ]
                        sim_check = pd.DataFrame(columns=sim_arr)
                        for date in row_datas.index:
                            if date not in cluster_index:
                                sim_check.loc[date] = [
                                    cos_sim(
                                        self.cluster_dict[idx -
                                                          (len(sim_arr) / 2)],
                                        row_datas.loc[date].values
                                    )
                                    if (len(sim_arr) / 2) <= idx else
                                    distance.euclidean(
                                        self.cluster_dict[idx],
                                        row_datas.loc[date].values
                                    )
                                    for idx in range(0, len(sim_arr))
                                ]
                        k_index = sim_check.sort_values(
                            by=sim_arr,
                            ascending=sim_sort_arr
                        ).index[0]
                        cluster_index.append(k_index)
                        self.cluster_dict[k] = row_datas.loc[k_index].copy()
                        self.cluster_cost_map[k] = 0

                    # return sim_arr, sim_sort_arr, cluster_dict

                # 두번째 클러스터링은 중심정을 찾아서
                else:
                    for k_num in self.cluster_dict.keys():
                        date_arr = self.cluster_info[
                            self.cluster_info['label'] == k_num
                        ].index
                        if len(date_arr) != 0:
                            self.cluster_dict[k_num] = row_datas.loc[date_arr].mean(
                            ).values
                        self.cluster_cost_map[k_num] = 0

                self.cluster_info = pd.DataFrame()
                self.visual_datas = pd.DataFrame()
                self.labels = []
                # print(self.cluster_dict)
                # print(type(self.cluster_dict[0]))
                # similar check
                cols = ['distance', 'similarity']
                rows = [idx for idx in range(0, self.K)]
                sim_info = pd.DataFrame(columns=cols)
                for date in row_datas.index:
                    sim_info = pd.DataFrame(columns=cols)

                    for row in rows:
                        sim_info.loc[row] = [
                            distance.euclidean(
                                self.cluster_dict[row],
                                row_datas.loc[date].values
                            ),
                            cos_sim(
                                self.cluster_dict[row],
                                row_datas.loc[date].values
                            )
                        ]
                    self.labels.append(sim_info.sort_values(
                        by=cols, ascending=[True, False]).index[0])

                # cluster info
                self.cluster_info['date'] = row_datas.index
                self.cluster_info['label'] = self.labels
                self.cluster_info.set_index('date', inplace=True)

                # visual data
                for date in row_datas.index:
                    tmp = pd.DataFrame()
                    tmp['timeslot'] = range(0, 24)
                    tmp['data'] = row_datas.loc[date].values
                    tmp['date'] = date
                    tmp['label'] = self.cluster_info.loc[date]['label']

                    self.visual_datas = pd.concat([
                        tmp,
                        self.visual_datas
                    ])

                sequence += 1
                print("TSS: {}, WSS: {}, ECV: {}".format(
                    self.get_TSS(), self.get_WSS(), self.get_ECV()))
                if prev_ecv == self.get_ECV():
                    break
                else:
                    prev_ecv = self.get_ECV()

            isInOne, findIdx = self.has_one_member()
            print("isInOne: {}, findIdx: {}".format(isInOne, findIdx))

            if self.get_ECV() >= 50:
                break

            if (len(findIdx) == 0) | (len(findIdx) >= (len(self.cluster_info['label'].unique()) - 2)):
                print("더 이상 제거할수가 없습니다.")
                print("-----COST MAP-----")
                print(self.cluster_cost_map)

                print("클러스터 분리를 시작합니다.")
                cost_map = pd.DataFrame(index=self.cluster_cost_map.keys())
                cost_map['count'] = self.cluster_info['label'].groupby(
                    self.cluster_info['label']).count()
                cost_map['cost'] = self.cluster_cost_map.values()

                print("영향 요소 ===> {}".format(
                    max(self.cluster_cost_map.items(), key=operator.itemgetter(1))[0]))

                test_1 = cost_map.sort_values(
                    by=["count", "cost"], ascending=False).index[0]
                test_2 = cost_map.sort_values(
                    by=["count"], ascending=[True]).index[0]

                if cost_map.loc[test_2]['count'] == 1:
                    test_2 = cost_map.sort_values(
                        by=["count", "cost"], ascending=[True, False]).index[0]
                print("Test Label", test_2)

                idxes = self.cluster_info[
                    self.cluster_info['label'] == test_1
                ].index

                new_dict_25 = pd.Series([np.percentile(
                    self.datas.loc[idx], 25) for idx in self.datas[idxes].index], name="cluster-divide-25")
                new_dict_75 = pd.Series([np.percentile(
                    self.datas.loc[idx], 75) for idx in self.datas[idxes].index], name="cluster-divide-75")
                self.cluster_cost_map[0] = 0
                self.cluster_cost_map[1] = 0

                self.remove_cluster_outlier([test_2])
                self.cluster_dict = {}
                self.cluster_dict[0] = new_dict_25
                self.cluster_dict[1] = new_dict_75

                divide_cluster = True
                # break
            else:
                divide_cluster = False
                self.remove_cluster_outlier(findIdx)

        cost_map = pd.DataFrame(index=self.cluster_cost_map.keys())
        cost_map['count'] = self.cluster_info['label'].groupby(
            self.cluster_info['label']).count()
        cost_map['cost'] = self.cluster_cost_map.values()
        return cost_map
        # self.run(ECV=ECV, remove_outlier=remove_outlier)
        # if ECV == True:
        #     break
        # 두번째 부터는 중심정 찾기

    # Best K Check
    def get_opt_K(self, init_condition="worst", ECV=False):
        sequence = 0
        row_datas = self.datas.T.copy()
        self.cluster_dict = {}
        prev_ecv = 0
        K = 2
        K_info = pd.DataFrame(columns=['k', 'ecv'])

        while True:
            while True:
                # 첫 클러스터링
                # Process , 첫 K 초기화
                if sequence == 0:
                    cluster_index = []
                    if init_condition == "random":
                        init_k = random.randint(0, len(row_datas.index) - 1)
                        init_cluster_index = row_datas.index[init_k]
                    elif init_condition == "best":
                        dr_datas = self.dimension_reduction()
                        init_cluster_index = dr_datas.sort_values(
                            ['x', 'y'], ascending=[True, False]).index[0]
                    else:
                        dr_datas = self.dimension_reduction()
                        init_cluster_index = dr_datas.sort_values(
                            ['x', 'y'], ascending=[False, True]).index[0]

                    init_cluster = row_datas.loc[init_cluster_index].copy()
                    cluster_index.append(init_cluster_index)
                    self.cluster_dict[0] = init_cluster
                    # 나머지 K 초기화
                    sim_arr = []
                    sim_sort_arr = []
                    for k in range(1, K):
                        sim_arr = ['dis_{}'.format(idx) for idx in range(0, k)]
                        sim_arr.extend(['cos_{}'.format(idx)
                                        for idx in range(0, k)])
                        sim_sort_arr = [
                            True if (len(sim_arr) / 2) <= idx else False
                            for idx in range(0, k * 2)
                        ]
                        sim_check = pd.DataFrame(columns=sim_arr)
                        for date in row_datas.index:
                            if date not in cluster_index:
                                sim_check.loc[date] = [
                                    cos_sim(
                                        self.cluster_dict[idx -
                                                          (len(sim_arr) / 2)],
                                        row_datas.loc[date].values
                                    )
                                    if (len(sim_arr) / 2) <= idx else
                                    distance.euclidean(
                                        self.cluster_dict[idx],
                                        row_datas.loc[date].values
                                    )
                                    for idx in range(0, len(sim_arr))
                                ]
                        k_index = sim_check.sort_values(
                            by=sim_arr,
                            ascending=sim_sort_arr
                        ).index[0]
                        cluster_index.append(k_index)
                        self.cluster_dict[k] = row_datas.loc[k_index].copy()

                    # return sim_arr, sim_sort_arr, cluster_dict

                # 두번째 클러스터링은 중심정을 찾아서
                else:
                    for k_num in self.cluster_dict.keys():
                        date_arr = self.cluster_info[
                            self.cluster_info['label'] == k_num
                        ].index
                        if len(date_arr) != 0:
                            self.cluster_dict[k_num] = row_datas.loc[date_arr].mean(
                            ).values

                self.cluster_info = pd.DataFrame()
                self.visual_datas = pd.DataFrame()
                self.labels = []

                # similar check
                cols = ['distance', 'similarity']
                rows = [idx for idx in range(0, K)]
                sim_info = pd.DataFrame(columns=cols)
                for date in row_datas.index:
                    sim_info = pd.DataFrame(columns=cols)

                    for row in rows:
                        sim_info.loc[row] = [
                            distance.euclidean(
                                self.cluster_dict[row],
                                row_datas.loc[date].values
                            ),
                            cos_sim(
                                self.cluster_dict[row],
                                row_datas.loc[date].values
                            )
                        ]
                    self.labels.append(sim_info.sort_values(
                        by=cols, ascending=[True, False]).index[0])

                # cluster info
                self.cluster_info['date'] = row_datas.index
                self.cluster_info['label'] = self.labels
                self.cluster_info.set_index('date', inplace=True)

                sequence += 1
                ECV = self.get_ECV()
                print("TSS: {}, WSS: {}, ECV: {}".format(
                    self.get_TSS(), self.get_WSS(), self.get_ECV()))
                if prev_ecv == ECV:
                    break
                else:
                    prev_ecv = ECV
                # if ECV == True:
                #     break
            print("--------K: {}, Maximum ECV: {}--------".format(K, self.get_ECV()))
            K_info = [
                K,
                self.get_ECV()
            ]
            if prev_ecv < 90:
                sequence = 0
                self.cluster_dict = {}
                prev_ecv = 0
                K += 1
            else:
                break
        return K_info

    def get_TSS(self):
        self.TSS = 0
        for date in self.datas:
            self.TSS += distance.euclidean(
                self.mean_pattern,
                self.datas[date].values
            ) ** 2
        return self.TSS

    def get_WSS(self):
        self.WSS = 0
        for date in self.datas:
            k_num = self.cluster_info.loc[date]['label']
            pattern = self.datas[date].values
            self.WSS += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2
            self.cluster_cost_map[k_num] += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2

        return self.WSS

    def get_ECV(self):
        return (1 - (self.get_WSS() / self.get_TSS())) * 100
