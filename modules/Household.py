import pandas as pd
from scipy.spatial import distance

SEASONS = {
    "봄": [3, 4, 5],
    "여름": [6, 7, 8],
    "가을": [9, 10, 11],
    "겨울": [12, 1, 2]
}

SEASONSARR = ['봄', '여름', '가을', '겨울']

'''
 0 : 봄
 1 : 여름
 2 : 가을
 3 : 겨울
'''

DAYARR = ["월", "화", "수", "목", "금", "토", "일"]


class Household:
    def __init__(self, uid, season_datas):
        self.uid = uid
        self.season_datas = season_datas
        self.merge_datas = {}

    def merging(self, hours=1):
        for season in SEASONS.keys():
            hours = 1
            merge_size = hours * 4
            merge_data = self.season_datas[season].copy()
            merge_data = merge_data[
                merge_data.columns.difference(['month', 'date'])
            ]
            merge_data.index = self.season_datas[season]['date']

            tmp = pd.DataFrame()
            start_idx = 0
            end_idx = 96
            for date in merge_data.index:
                in_arr = []
                og_arr = merge_data.loc[date].values
                for merging in range(0, int(end_idx / merge_size)):
                    in_arr.append(og_arr[
                        (merging) * merge_size:
                        (merging + 1) * merge_size
                    ].sum())
            #     logging
            #     print("date end : {}, size: {}".format(date, len(in_arr)))

                tmp[date] = in_arr
            merge_data = tmp.copy()

            self.merge_datas[season] = merge_data.copy()

    def TSS(self, season='봄'):
        mean_pattern = self.merge_datas[season].T.mean()
        TSS = 0

        for date in self.merge_datas[season]:
            TSS += distance.euclidean(
                mean_pattern,
                self.merge_datas[season][date].values,
            ) ** 2

        return TSS
