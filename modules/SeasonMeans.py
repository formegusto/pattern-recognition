import operator
from scipy.spatial import distance
import random

import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B)/(norm(A) * norm(B))


def min_max_normalization(list):
    return [
        (val - list.min()) /
        (list.max() - list.min())
        for val in
        list.values
    ]


class SeasonMeans:
    def __init__(self, init_datas):
        print("---Season Means---")
        self.datas = init_datas
        self.mean_pattern = self.datas.mean()
        self.cluster_dict = {}
        self.cluster_info = pd.DataFrame(columns=['label'])

    def calc_tss(self):
        self.tss = 0
        for index in self.datas.index:
            self.tss += distance.euclidean(
                self.mean_pattern,
                self.datas.loc[index].values
            ) ** 2

        print("- calc TSS(Total Sum Of Squares) success!")

    def calc_wss(self):
        self.wss = 0
        for index in self.datas.index:
            k_num = self.cluster_info.loc[index]['label']
            pattern = self.datas.loc[index].values
            self.wss += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2
        print("- calc WSS success!")

    def cals_ecv(self):
        self.ecv = (1 - (self.wss) / self.tss) * 100
        print("- calc ECV success!")

    def dimension_reduction(self):
        print("---Dimension Reduction---")

        index = pd.MultiIndex.from_tuples((), names=['uid', 'date'])
        dr_datas = pd.DataFrame(columns=['x', 'y'], index=index)
        for data in self.datas.index:
            uid, date = data
            dr_datas.loc[(uid, date), :] = [
                distance.euclidean(self.mean_pattern,
                                   self.datas.loc[(uid, date)]),
                cos_sim(self.mean_pattern, self.datas.loc[(uid, date)])
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

        og_length = len(self.datas.index)
        self.dr_datas = self.dr_datas.loc[~self.dr_datas.index.isin(
            remove_index)]
        self.datas = self.datas.loc[~self.datas.index.isin(remove_index)]

        new_length = len(self.datas.index)
        print("- remove outlier success: {} => {}".format(og_length, new_length))
        self.calc_tss()

    def run(self, K=10):
        print("---init TSS Check---")
        self.calc_tss()

        self.dimension_reduction()
        self.remove_outlier()

        sequence = 0
        prev_ecv = 0
        print("---{}:Clustering Start---".format(K))
        self.cluster_dict = {}

        while True:
            print("---Now {}---".format(1))
            dr_datas = self.dr_datas.copy()
            datas = self.datas.copy()

            if sequence == 0:
                print("---First Cluster Group Init---")
                print("---First K Select---")
                cluster_index = []
                k_uid, k_date = dr_datas.sort_values(
                    ['x', 'y'], ascending=[False, True]).index[0]
                cluster_index.append("{}/{}".format(k_uid, k_date))
                init_cluster = datas.loc[(k_uid, k_date)].values
                self.cluster_dict[0] = init_cluster
                print("- First K Is {}".format((k_uid, k_date)))

                print("---Rest K Select---")
                for k in range(len(self.cluster_dict.keys()), K):
                    sim_arr = ['dis_{}'.format(idx) for idx in range(0, k)]
                    sim_arr.extend(['cos_{}'.format(idx)
                                   for idx in range(0, k)])
                    sim_sort_arr = [
                        True if (len(sim_arr) / 2) <= idx else False
                        for idx in range(0, k * 2)
                    ]
                    index = pd.MultiIndex.from_tuples(
                        (), names=['uid', 'date'])
                    sim_check = pd.DataFrame(columns=sim_arr, index=index)
                    for uid, date in datas.index:
                        if "{}/{}".format(uid, date) not in cluster_index:
                            sim_check.loc[(uid, date), :] = [
                                cos_sim(
                                    self.cluster_dict[idx -
                                                      (len(sim_arr) / 2)],
                                    datas.loc[(uid, date)].values
                                )
                                if (len(sim_arr) / 2) <= idx else
                                distance.euclidean(
                                    self.cluster_dict[idx],
                                    datas.loc[(uid, date)].values,
                                )
                                for idx in range(0, len(sim_arr))
                            ]
                    k_uid, k_date = sim_check.sort_values(
                        by=sim_arr,
                        ascending=sim_sort_arr
                    ).index[0]
                    cluster_index.append("{}/{}".format(k_uid, k_date))
                    self.cluster_dict[k] = datas.loc[(
                        k_uid, k_date)].values.copy()
                    print("- {} K Is {}".format(k + 1,
                                                cluster_index[len(cluster_index) - 1]))
            else:
                for k_num in self.cluster_dict.keys():
                    idx_arr = self.cluster_info[
                        self.cluster_info['label'] == k_num
                    ].index
                    if len(idx_arr) != 0:
                        self.cluster_dict[k_num] = self.datas.loc[idx_arr].mean(
                        ).values

            # Clustering
            print("---Cluster Init Okay KMeans Start---")
            index = pd.MultiIndex.from_tuples((), names=['uid', 'date'])
            self.cluster_info = pd.DataFrame()
            self.visual_datas = pd.DataFrame()
            self.labels = []

            cols = ['distance', 'similarity']
            rows = [idx for idx in range(0, K)]
            for uid, date in datas.index:
                sim_info = pd.DataFrame(columns=cols)
                for row in rows:
                    sim_info.loc[row] = [
                        distance.euclidean(
                            self.cluster_dict[row],
                            datas.loc[(uid, date)].values
                        ),
                        cos_sim(
                            self.cluster_dict[row],
                            datas.loc[(uid, date)].values
                        )
                    ]
                self.labels.append(sim_info.sort_values(
                    by=cols, ascending=[True, False]).index[0])

            # cluster info
            self.cluster_info['uid'] = [uid for uid, date in datas.index]
            self.cluster_info['date'] = [date for uid, date in datas.index]
            self.cluster_info['label'] = self.labels
            self.cluster_info.set_index(['uid', 'date'], inplace=True)

            # visual data
            for uid, date in datas.index:
                tmp = pd.DataFrame()
                tmp['timeslot'] = range(0, 24)
                tmp['uid'] = uid
                tmp['date'] = date
                tmp['data'] = list(datas.loc[(uid, date)].values)
                tmp['label'] = self.cluster_info.loc[(uid, date)]['label']

                self.visual_datas = pd.concat([
                    tmp,
                    self.visual_datas
                ], ignore_index=True)
            sequence += 1

            self.calc_wss()
            self.cals_ecv()
            print("TSS: {}, WSS: {}, ECV: {}".format(
                self.tss,
                self.wss,
                self.ecv
            ))

            if prev_ecv == self.wss:
                break
            else:
                prev_ecv = self.wss
                continue
