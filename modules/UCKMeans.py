import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance


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


class UCKMeans:
    def __init__(self, init_datas, K=2):
        print("---UCKMeans Init---\nK:{}".format(K))
        self.datas = init_datas
        self.cluster_dict = {}
        self.labels = []
        self.cluster_info = pd.DataFrame()
        self.mean_pattern = self.datas.T.mean()
        self.distance_map = pd.DataFrame(columns=['distance'])
        for date in self.datas:
            self.distance_map.loc[date] = [
                distance.euclidean(
                    self.mean_pattern,
                    self.datas[date].values
                )
            ]
        
        self.distance_map['distance'] = min_max_normalization(self.distance_map['distance'])
            
        self.K = K

    def dimension_reduction(self):
        dr_datas = pd.DataFrame(columns=['x', 'y'])

        for date in self.datas.copy():
            dr_datas.loc[date] = [
                distance.euclidean(
                    self.mean_pattern,
                    self.datas[date].values
                ),
                cos_sim(
                    self.mean_pattern,
                    self.datas[date].values
                )
            ]

        # dr_datas['x'] = min_max_normalization(dr_datas['x'])
        # dr_datas['y'] = min_max_normalization(dr_datas['y'])

        self.dr_mean_pattern = [
            dr_datas['x'].mean(),
            dr_datas['y'].mean(),
        ]
        self.dr_datas = dr_datas.copy()

        return dr_datas

    def remove_one_pattern(self):
        remove_idxes = []
        for idx in self.datas.copy():
            if len(set(self.datas[idx].values)) == 1:
                remove_idxes.append(idx)

        og_length = len(self.datas.columns)

        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_idxes)]
        new_datas = new_datas.T

        self.datas = new_datas.copy()
        self.dr_datas = self.dimension_reduction()
        print(remove_idxes)
        print("""-------Remove One Pattern Success-------\n  Target length {} -> {}""".format(og_length,
              len(self.datas.columns)))

    def remove_outlier(self):
        datas = self.distance_map.copy()
        outlier_range = 1.5

        dis_info = {
            "Q1": np.percentile(datas['distance'], 25),
            "Q3": np.percentile(datas['distance'], 75),
        }
        dis_info["IQR"] = dis_info["Q3"] - dis_info["Q1"]
        dis_info["step"] = outlier_range * dis_info["IQR"]
        print("DIS_INFO", dis_info)
        remove_idxes = datas[
            (datas['distance'] > dis_info["Q3"] + dis_info['step'])
        ].index

        og_length = len(self.datas.columns)
        new_datas = self.datas.copy()
        new_datas = new_datas.T
        new_datas = new_datas.loc[~new_datas.index.isin(remove_idxes)]
        new_datas = new_datas.T
        self.datas = new_datas.copy()
        print(remove_idxes)
        # new distance map
        self.distance_map = pd.DataFrame(columns=['distance'])
        for date in self.datas:
            self.distance_map.loc[date] = [
                distance.euclidean(
                    self.mean_pattern,
                    self.datas[date].values
                )
            ]
        self.mean_pattern = self.datas.T.mean()

        print("""-------Remove Outlier Success-------\n  Target length {} -> {}""".format(og_length,
              len(self.datas.columns)))

    def run(self):
        print("---Go UCMeans---")
        self.remove_one_pattern()
        self.remove_outlier()

        sequence = 0
        prev_ecv = 0
        while True:
            self.visual_datas = pd.DataFrame()
            print("---Now {}---".format(sequence))
            # ??? ???????????????
            # ??? K ??????
            # ????????? ?????????
            if sequence == 0:
                # print("First K Init!!")
                cluster_index = []
                # ??? K ??????
                init_cluster_idx = self.distance_map.sort_values(
                    by="distance", ascending=False).index[0]
                cluster_index.append(init_cluster_idx)
                init_clutser_pattern = self.datas[init_cluster_idx]
                self.cluster_dict[0] = init_clutser_pattern

                # print("rest K Init!!")
                for k in range(len(self.cluster_dict.keys()), self.K):
                    dis_arr = ['dis_{}'.format(idx) for idx in range(0, k)]
                    dis_check = pd.DataFrame(columns=dis_arr)
                    for date in self.datas:
                        if date not in cluster_index:
                            dis_check.loc[date] = [
                                distance.euclidean(
                                    self.cluster_dict[idx],
                                    self.datas[date].values
                                ) for idx in range(0, len(dis_arr))
                            ]
                    k_idx = dis_check.sort_values(
                        by=dis_arr,
                        ascending=False
                    ).index[0]
                    cluster_index.append(k_idx)
                    self.cluster_dict[k] = self.datas[k_idx].copy()

            else:
                for k_num in self.cluster_dict.keys():
                    date_arr = self.cluster_info[
                        self.cluster_info['label'] == k_num
                    ].index
                    if len(date_arr) != 0:
                        self.cluster_dict[k_num] = self.datas.T.loc[date_arr].mean(
                        ).values

            self.cluster_info = pd.DataFrame()
            self.visual_datas = pd.DataFrame()
            self.labels = []

            for date in self.datas:
                dis_info = pd.DataFrame(columns=["distance"])

                for row in range(self.K):
                    dis_info.loc[row] = [
                        distance.euclidean(
                            self.cluster_dict[row],
                            self.datas[date].values
                        )
                    ]
                # print(dis_info)
                # print("set: {}".format(dis_info.sort_values(
                #     by=["distance"],
                #     ascending=True
                # ).index[0]))
                self.labels.append(dis_info.sort_values(
                    by=["distance"],
                    ascending=True
                ).index[0])

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

            print("TSS: {}, WSS: {}, ECV: {}, CPDV: {}".format(
                    self.get_TSS(), self.get_WSS(), self.get_ECV(), self.get_CPDV()))
            # print("UCOSTSS: {}, UCOSWSS: {}, UCOSECV: {}".format(
            #     self.get_UCOSTSS(), self.get_UCOSWSS(), self.get_UCOSECV()))
            if prev_ecv == self.get_ECV():
                # print("TSS: {}, WSS: {}, ECV: {}, CPDV: {}".format(
                #     self.get_TSS(), self.get_WSS(), self.get_ECV(), self.get_CPDV()))
                break
            else:
                prev_ecv = self.get_ECV()
            sequence += 1

    def get_TSS(self):
        self.TSS = 0
        for date in self.datas:
            self.TSS += distance.euclidean(
                self.mean_pattern,
                self.datas[date].values
            ) ** 2

        return self.TSS

    def get_UCOSTSS(self):
        self.UCOSTSS = 0
        dr_datas = self.dimension_reduction()
        for date in self.datas:
            self.UCOSTSS += distance.euclidean(
                self.dr_mean_pattern,
                dr_datas.loc[date].values
            ) ** 2
        return self.UCOSTSS

    def get_WSS(self):
        self.WSS = 0
        for date in self.datas:
            k_num = self.cluster_info.loc[date]['label']
            pattern = self.datas[date].values
            self.WSS += distance.euclidean(
                pattern,
                self.cluster_dict[k_num]
            ) ** 2

        return self.WSS

    def get_UCOSWSS(self):
        self.UCOSWSS = 0
        dr_datas = self.dimension_reduction()

        for date in self.datas:
            k_num = self.cluster_info.loc[date]['label']
            # print(self.cluster_dict[k_num])
            cluster_dr_datas = [
                distance.euclidean(
                    self.mean_pattern,
                    self.cluster_dict[k_num]
                ),
                cos_sim(
                    self.mean_pattern,
                    self.cluster_dict[k_num]
                )
            ]
            pattern = dr_datas.loc[date].values
            # print("??????: {}".format(date), cluster_dr_datas, pattern)
            self.UCOSWSS += distance.euclidean(
                cluster_dr_datas,
                pattern
            ) ** 2

        return self.UCOSWSS

    def get_ECV(self):
        return (1 - (self.get_WSS() / self.get_TSS())) * 100

    def get_UCOSECV(self):
        return (1 - self.get_UCOSWSS() / self.get_UCOSTSS()) * 100

    # Cluster Pattern Direction Value
    def get_CPDV(self):
        self.CPDV = 0
        tmp_cpdv = []
        for k_num in self.cluster_dict:
            index = self.cluster_info[
                self.cluster_info['label'] == k_num
            ].index
            for date in index:
                tmp_cpdv.append(
                    cos_sim(
                        self.cluster_dict[k_num],
                        self.datas[date].values
                    )
                )
        self.CPDV = np.array(tmp_cpdv).mean()
        return self.CPDV
