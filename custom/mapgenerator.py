import pandas as pd


def get_map_datas(datas, general_datas):
    check_str = ''

    if '월' in datas:
        check_str = '월'
    else:
        check_str = '봄'

    if type(datas[check_str]['date'][0]) != type(general_datas.columns[0]):
        general_datas = general_datas.T.copy()
    map_datas = pd.DataFrame()

    for date in general_datas:
        new = pd.DataFrame()
        new['timeslot'] = range(0, 96)
        new['data'] = general_datas[date].values
        new['date'] = date
        map_datas = pd.concat([map_datas, new])

    return map_datas


def get_map_datas_cluster(datas, general_datas, cluster_datas):
    if type(datas['봄']['date'][0]) != type(general_datas.columns[0]):
        general_datas = general_datas.T.copy()
    map_datas = pd.DataFrame()

    for date in general_datas:
        new = pd.DataFrame()
        new['timeslot'] = range(0, 96)
        new['data'] = general_datas[date].values
        new['date'] = date
        new['cluster'] = cluster_datas[date]['cluster']
        map_datas = pd.concat([map_datas, new])

    map_datas['date'] = pd.to_datetime(map_datas['date'])

    return map_datas
