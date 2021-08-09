import operator
from scipy.spatial import distance
import random

import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm

import math


def cos_sim(A, B):
    return dot(A, B)/(norm(A) * norm(B))


def min_max_normalization(list):
    return [
        (val - list.min()) /
        (list.max() - list.min())
        for val in
        list.values
    ]


class THMeans:
    def __init__(self, init_datas):
        print("---Season Means---")
        self.datas = init_datas
        self.mean_pattern = self.datas.T.mean()
        self.cluster_dict = {}
        self.cluster_info = pd.DataFrame(columns=['label'])

    def calc_tss(self):
        self.tss = 0
        for date in self.datas:
            self.tss += distance.euclidean(
                self.mean_pattern,
                self.datas[date].values
            ) ** 2

        print("- calc TSS(Total Sum Of Squares) success!")

    def calc_wss(self):
        self.wss = 0
        for date in self.datas:
            k_num = self.cluster_info.loc[date]['label']
            pattern = self.datas[date].values
            self.wss += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2
        print("- calc WSS(Within cluster Sum Of Squares) success!")

    def calc_ecv(self):
        self.ecv = (1 - (self.wss) / self.tss) * 100
        print("- calc ECV(Explained Cluster Value) success!")

    def calc_cdpv(self):
        self.cdpv = 0
        tmp_cdpv = []
        for k_num in self.cluster_dict:
            index = self.cluster_info[
                self.cluster_info['label'] == k_num
            ].index
            for date in index:
                tmp_cdpv.append(
                    cos_sim(
                        self.cluster_dict[k_num],
                        self.datas[date].values
                    )
                )
        self.cdpv = np.array(tmp_cdpv).mean()
        print("- calc CDPV(Cluster Pattern Direction Value) success!")

    def dimension_reduction(self):
        print("---Dimension Reduction---")

        dr_datas = pd.DataFrame(columns=['x', 'y'])
        for data in self.datas:
            date = data
            dr_datas.loc[date] = [
                distance.euclidean(self.mean_pattern,
                                   self.datas[date]),
                cos_sim(self.mean_pattern, self.datas[date])
            ]

        dr_datas['x'] = min_max_normalization(dr_datas['x'])
        dr_datas['y'] = min_max_normalization(dr_datas['y'])

        self.dr_datas = dr_datas
        return dr_datas

    def remove_outlier(self):
        print("---Remove Outliers---")
        outlier_range = 1.5
        dis_check = np.percentile(
            self.dr_datas['x'], 75) + (np.percentile(
                self.dr_datas['x'], 75) - np.percentile(self.dr_datas['x'], 25)) * outlier_range
        sim_check = np.percentile(
            self.dr_datas['y'], 25) - (np.percentile(
                self.dr_datas['y'], 75) - np.percentile(self.dr_datas['y'], 25)) * outlier_range

        print("- dis_check: {}, sim_check: {}".format(dis_check, sim_check))

        remove_index = self.dr_datas[
            (self.dr_datas['x'] >= dis_check) |
            (self.dr_datas['y'] <= sim_check)
        ].index

        self.og_length = len(self.datas.columns)

        self.dr_datas = self.dr_datas.loc[~self.dr_datas.index.isin(
            remove_index)]

        self.mdis = self.dr_datas['x'].mean()
        self.mcdpv = self.dr_datas['y'].mean()

        self.datas = self.datas.loc[:, ~self.datas.columns.isin(remove_index)]

        self.new_length = len(self.datas.columns)
        print("- remove outlier success: {} => {}".format(self.og_length, self.new_length))
        self.calc_tss()

    def remove_one_pattern(self):
        remove_idxes = []
        for idx in self.datas.copy():
            if len(set(self.datas[idx].values)) == 1:
                remove_idxes.append(idx)

        self.og_length = len(self.datas.columns)

        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_idxes)]
        new_datas = new_datas.T

        self.datas = new_datas.copy()
        self.dr_datas = self.dimension_reduction()

        print(remove_idxes)

    def get_divide_index(self, K, length):
        print("Test Start!")
        # 홀수 배열 테스트
        mean = 0
        arr = [idx for idx in range(0, length)]
        print("arr length: {}".format(len(arr)))

        # 가장 첫번째 (평균)
        # 가장 먼 데이터
        print("Cluster Worst Pattern In")
        idxes = [0, arr[length - 1]]
        # K
        d_weight = 1
        d_weight_sum = 1
        while True:
            # 내가 여기에 숫자 인덱스를 담을거야
            # 홀수개면 마지막 인덱스는 짝수
            # print(idxes)
            tmp = idxes.copy()
            tmp.sort()
            tmp.reverse()

            # print(tmp)

            for dseq in range(0, d_weight):
                print("{}: calc... {}".format(d_weight, tmp))
                in_idx = math.ceil(tmp[dseq] / 2)
                if len(idxes) != (dseq + 1):
                    in_idx = math.ceil((tmp[dseq] + tmp[dseq + 1]) / 2)
                tmp.append(in_idx)

            idxes = tmp.copy()
            print("{}: {}".format(d_weight, idxes))
            if len(idxes) >= K:
                idxes = idxes[0: K]
                break
            else:
                d_weight += d_weight_sum
                d_weight_sum += 1
        idxes.remove(0)
        return idxes

    def init_cluster(self):
        print("---Init Cluster---")

        # 초기 패턴은 mean_pattern 설정
        print("---First K Is Mean Pattern---")
        self.cluster_dict[0] = self.mean_pattern

        sort_dr_datas = self.dr_datas.sort_values(
            ['x', 'y'], ascending=[False, True]).copy()
        print("---Rest K Select---")
        idxes = self.get_divide_index(self.K, len(sort_dr_datas.index))
        for k in range(1, self.K):
            for idx in idxes:
                self.cluster_dict[len(
                    self.cluster_dict.keys()
                )] = self.datas.iloc[:, idx].values

            print("-{}: K Setting Okay".format(k + 1))

    def run(self, K=10):
        print("---init TSS Check---")
        self.calc_tss()

        self.remove_one_pattern()
        self.remove_outlier()

        self.sequence = 1
        prev_ecv = 0

        self.K = round(math.sqrt((len(self.datas.columns) / 2)))
        K = self.K
        print("---K Setting {} ---".format(K))

        print("---{}:Clustering Start---".format(K))
        self.cluster_dict = {}

        while True:
            print("---Now {}---".format(self.sequence))
            datas = self.datas.copy()

            if self.sequence == 1:
                print("---First Cluster Group Init---")
                self.init_cluster()
            else:
                for k_num in self.cluster_dict.keys():
                    idx_arr = self.cluster_info[
                        self.cluster_info['label'] == k_num
                    ].index.values
                    if len(idx_arr) != 0:
                        self.cluster_dict[k_num] = self.datas[idx_arr].T.mean(
                        ).values
            # Clustering
            print("---Cluster Init Okay KMeans Start---")
            self.cluster_info = pd.DataFrame()
            self.visual_datas = pd.DataFrame()
            self.labels = []

            cols = ['distance', 'similarity']
            rows = [idx for idx in range(0, K)]
            for date in datas:
                sim_info = pd.DataFrame(columns=cols)
                for row in rows:
                    sim_info.loc[row] = [
                        distance.euclidean(
                            self.cluster_dict[row],
                            datas[date].values
                        ),
                        cos_sim(
                            self.cluster_dict[row],
                            datas[date].values
                        )
                    ]
                self.labels.append(sim_info.sort_values(
                    by=cols, ascending=[True, False]).index[0])

            # cluster info
            self.cluster_info['date'] = datas.columns
            self.cluster_info['label'] = self.labels
            self.cluster_info.set_index(['date'], inplace=True)

            # visual data
            for date in datas:
                tmp = pd.DataFrame()
                tmp['timeslot'] = range(0, 24)
                tmp['date'] = date
                tmp['data'] = list(datas[date].values)
                tmp['label'] = self.cluster_info.loc[date]['label']

                self.visual_datas = pd.concat([
                    tmp,
                    self.visual_datas
                ], ignore_index=True)

            self.calc_wss()
            self.calc_ecv()
            self.calc_cdpv()
            print("{} : TSS: {}, WSS: {}, ECV: {}, CDPV: {}".format(
                self.sequence,
                self.tss,
                self.wss,
                self.ecv,
                self.cdpv
            ))

            if prev_ecv == self.wss:
                break
            else:
                self.sequence += 1
                prev_ecv = self.wss
                continue
