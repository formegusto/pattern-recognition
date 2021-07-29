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


class UCOSKMeans:
    def __init__(self, init_datas, K=2):
        print("---UCOSKMeans Init---\nK:{}".format(K))

        self.K = K
        self.datas = init_datas
        self.mean_pattern = self.datas.T.mean()
        self.labels = []
        self.cluster_dict = {}
        self.cluster_info = pd.DataFrame()
        self.visual_datas = pd.DataFrame()
        self.dr_datas = pd.DataFrame()

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

        self.dr_mean_pattern = [
            dr_datas['x'].mean(),
            dr_datas['y'].mean()
        ]

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
        print("DIS_INFO", dis_info)

        sim_info = {
            "Q1": np.percentile(datas['y'], 25),
            "Q3": np.percentile(datas['y'], 75)
        }
        sim_info["IQR"] = sim_info["Q3"] - sim_info["Q1"]
        sim_info["step"] = outlier_range * sim_info["IQR"]
        sim_info["check"] = sim_info["Q1"] - sim_info["step"]
        print("SIM_INFO", sim_info)

        remove_outlier_index = datas[
            ((datas['x'] > dis_info['check'])
             |
             (datas['y'] < sim_info['check']))
        ].copy().index

        # # 1자 멤버 추가적으로 제거
        # print([col if len(set(self.datas[col].values))
        #       == 1 else None for col in self.datas])

        og_length = len(self.datas.columns)

        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_outlier_index)]
        new_datas = new_datas.T

        self.datas = new_datas.copy()
        self.dr_datas = self.dimension_reduction()

        print("""-------Remove Outlier Success-------\n  Target length {} -> {}""".format(og_length,
              len(self.datas.columns)))

        return new_datas, datas

    def run(self):
        print("---Go UCOSMeans---")
        self.remove_outlier()

        sequence = 0
        prev_ecv = 0
        while True:
            self.visual_datas = pd.DataFrame()
            print("---Now {}---".format(sequence))

            if sequence == 0:
                print("First K Init")
                cluster_index = []
                # 첫 K 선정
                init_cluster_idx = self.dr_datas.sort_values(
                    ['x', 'y'], ascending=[False, True]).index[0]

                init_cluster = self.dr_datas.loc[init_cluster_idx]
                cluster_index.append(init_cluster_idx)
                self.cluster_dict[0] = init_cluster

                print("rest K Init!!")
                for k in range(len(self.cluster_dict.keys()), self.K):
                    sim_arr = ['dis_{}'.format(idx) for idx in range(0, k)]
                    # sim_arr.extend(['cos_{}'.format(idx)
                    #                 for idx in range(0, k)])
                    # sim_sort_arr = [
                    #     True if (len(sim_arr) / 2) <= idx else False
                    #     for idx in range(0, k * 2)
                    # ]
                    sim_check = pd.DataFrame(columns=sim_arr)
                    for date in self.datas:
                        if date not in cluster_index:
                            sim_check.loc[date] = [
                                distance.euclidean(
                                    self.cluster_dict[idx],
                                    self.dr_datas.loc[date].values
                                )
                                for idx in range(0, len(sim_arr))
                            ]
                    k_index = sim_check.sort_values(
                        by=sim_arr,
                        ascending=False
                    ).index[0]
                    cluster_index.append(k_index)
                    self.cluster_dict[k] = self.dr_datas.loc[k_index].copy()
            else:
                for k_num in self.cluster_dict.keys():
                    date_arr = self.cluster_info[
                        self.cluster_info['label'] == k_num
                    ].index
                    if len(date_arr) != 0:
                        self.cluster_dict[k_num] = self.dr_datas.loc[date_arr].mean(
                        ).values
            self.cluster_info = pd.DataFrame()
            self.visual_datas = pd.DataFrame()
            self.labels = []

            cols = ['distance']
            rows = [idx for idx in range(0, self.K)]
            sim_info = pd.DataFrame(columns=cols)
            for date in self.datas:
                sim_info = pd.DataFrame(columns=cols)

                for row in rows:
                    sim_info.loc[row] = [
                        distance.euclidean(
                            self.cluster_dict[row],
                            self.dr_datas.loc[date].values
                        )
                    ]
                self.labels.append(sim_info.sort_values(
                    by=cols, ascending=True).index[0])
            # cluster info
            self.cluster_info['date'] = self.datas.columns
            self.cluster_info['label'] = self.labels
            self.cluster_info.set_index('date', inplace=True)

            # visual data
            for date in self.datas:
                tmp = pd.DataFrame()
                tmp['timeslot'] = range(0, 24)
                tmp['data'] = self.datas[date].values
                tmp['date'] = date
                tmp['label'] = self.cluster_info.loc[date]['label']

                self.visual_datas = pd.concat([
                    tmp,
                    self.visual_datas
                ])
            sequence += 1

            print("TSS: {}, WSS: {}, ECV: {}".format(
                self.get_UCTSS(), self.get_UCWSS(), self.get_UCECV()))
            print("UCOSTSS: {}, UCOSWSS: {}, UCOSECV: {}".format(
                self.get_TSS(), self.get_WSS(), self.get_ECV()))

            if prev_ecv == self.get_ECV():
                break
            else:
                prev_ecv = self.get_ECV()

    def get_UCTSS(self):
        self.UCTSS = 0
        for date in self.datas:
            self.UCTSS += distance.euclidean(
                self.datas[date].values,
                self.mean_pattern
            ) ** 2
        return self.UCTSS

    def get_UCWSS(self):
        self.UCWSS = 0

        # 데이터 구축
        cluster_patterns = {}
        for k_num in self.cluster_dict.keys():
            num, dates = k_num, self.cluster_info[
                self.cluster_info['label'] == k_num
            ].index
            cluster_patterns[k_num] = self.datas.T.loc[dates].mean().values

        for date in self.cluster_info.index:
            k_num = self.cluster_info.loc[date]['label']
            self.UCWSS += distance.euclidean(
                cluster_patterns[k_num],
                self.datas[date].values
            ) ** 2

        return self.UCWSS

    def get_UCECV(self):
        return (1 - (self.get_UCWSS() / self.get_UCTSS())) * 100

    def get_TSS(self):
        self.TSS = 0
        for date in self.datas:
            self.TSS += distance.euclidean(
                self.dr_mean_pattern,
                self.dr_datas.loc[date].values
            ) ** 2
        return self.TSS

    def get_WSS(self):
        self.WSS = 0
        for date in self.datas:
            k_num = self.cluster_info.loc[date]['label']
            pattern = self.dr_datas.loc[date].values
            self.WSS += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2

        return self.WSS

    def get_ECV(self):
        return (1 - (self.get_WSS() / self.get_TSS())) * 100
