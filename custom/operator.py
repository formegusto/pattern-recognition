import pandas as pd
from scipy.spatial import distance


def get_ECV(all_datas, cluster_map_datas):
    # ECV 공식
    # 필요변수
    # mean, TSS(Total Sum of Squares), WSS(Within cluster Sum of Squares)
    # mean = 전체 데이터 평균
    # TSS = 전체 데이터 timeslot과 평균의 timeslot 거리 ^ 2
    # WSS = 전체 데이터 timeslot과 전체 데이터의 자신의 main cluster와의 거리
    # 필요한 거 : 전체 데이터 timeslot 평균 패턴 ,cluster 패턴 데이터

    # 전체 데이터 timeslot 평균 패턴
    mean_pattern = []
    for ts_idx in range(0, 96):
        mean_pattern.append(cluster_map_datas['data'][ts_idx].mean())
    mean_pattern

    # cluster 패턴 데이터
    cluster_info = pd.DataFrame()
    for cluster_num in cluster_map_datas['cluster'].unique():
        cluster_tmp = pd.DataFrame()
        tmp = cluster_map_datas[cluster_map_datas['cluster']
                                == cluster_num][['data']].copy()

        cluster_tmp['data'] = [tmp['data'][idx].mean() for idx in range(0, 96)]
        cluster_tmp['label'] = cluster_num
        cluster_tmp['timeslot'] = [idx for idx in range(0, 96)]

        cluster_info = pd.concat([cluster_info, cluster_tmp])

    # TSS
    TSS = 0
    for uid, date in all_datas.index:
        if len(cluster_map_datas[
            (cluster_map_datas['uid'] == uid) &
            (cluster_map_datas['date'] == date)
        ]) == 0:
            continue
        houself_pattern = all_datas.loc[(uid, date)].values
        TSS += distance.euclidean(mean_pattern,
                                  houself_pattern) ** 2

    # WSS
    WSS = 0
    for data_idx in all_datas.index:
        pattern_arr = cluster_map_datas[
            (cluster_map_datas['uid'] == data_idx[0]) &
            (cluster_map_datas['date'] == data_idx[1])
        ]['data'].values

        if len(pattern_arr) == 0:
            continue

        main_cluster = cluster_map_datas[
            (cluster_map_datas['uid'] == data_idx[0]) &
            (cluster_map_datas['date'] == data_idx[1])
        ]['cluster'][0]

        WSS += distance.euclidean(pattern_arr,
                                  cluster_info[
                                      cluster_info['label'] == main_cluster
                                  ]['data'].values
                                  ) ** 2

    ECV = 100 * (TSS-WSS) / TSS
    print("TSS: {}, WSS: {}".format(TSS, WSS))
    print("ECV: {}".format(ECV))
