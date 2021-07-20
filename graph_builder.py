import os
import json
import random
import datetime
import requests
import numpy as np
import pandas as pd

import networkx as nx
from impala.dbapi import connect


class GraphBuilder:
    def __init__(self, data_dir, seed=12345):#, accident_api=None):#, speed_api=None):
        self.data_dir = data_dir
        random.seed = seed

        self.accident_api = 'http://218.150.247.209:52201/link' # http://10.100.20.61:{port}/link
        self.speed_api = 'https://openapi.its.go.kr:9443/trafficInfo'
        self.speed_api_flag = True          # True: api, False: DB

        self.area_coords = [127.2575497, 127.3446409, 36.4603768, 36.5116487]


    def build_graph(self):
        stop_nodes = [
            4130022800, 4130026600, 4130023200, 4130023000, 4130026500, 4130022900, 4130105600, 4130109800, 4130022000,
        ]
        stop_links = [
            4130191300, 4130191400, 4130148300, 4130148400, 4130126900, 4130126300, 
        ]

        df_sejong_nodes = pd.read_csv(
            os.path.join(self.data_dir, 'AUTO_AREA_NODE.csv'), encoding='cp949')

        sejong_node_ids = df_sejong_nodes['NODE_ID'].to_list()

        np_link_info = np.load(os.path.join(self.data_dir, 'LINK_INFO.npy'))
        df_link = pd.DataFrame(np_link_info)
        df_link.columns = [
            "LINK_ID", "F_NODE", "T_NODE", "LANES",
            "ROAD_RANK", "ROAD_TYPE", "ROAD_NO", "ROAD_NAME",
            "ROAD_USE", "MULTI_LINK", "CONNECT", "MAX_SPD",
            "REST_VEH", "REST_W", "REST_H", "LENGTH",
            "VERTEX_CNT", "REMARK"
        ]

        np_node_info = np.load(os.path.join(self.data_dir, 'NODE_INFO.npy'))
        df_node_info = pd.DataFrame(np_node_info)
        df_node_info.columns = [
            "NODE_ID","NODE_TYPE","NODE_NAME","TURN_P","REMARK"
        ]
        np_node_GIS = np.load(os.path.join(self.data_dir, 'NODE_GIS.npy'))
        df_node_GIS = pd.DataFrame(np_node_GIS)
        df_node_GIS.columns = ["NODE_ID","LONGITUDE","LATITUDE","ELEVATION"]
        df_node = pd.concat((df_node_info,df_node_GIS[["LONGITUDE","LATITUDE","ELEVATION"]]), axis=1)

        G = nx.DiGraph()

        # node
        for (i, row) in df_node.iterrows():
            if int(row['NODE_ID']) in sejong_node_ids:
                if int(row['NODE_ID']) in stop_nodes:
                    continue
                G.add_node(int(row['NODE_ID']), coordinate=(row['LONGITUDE'], row['LATITUDE']))

        # link
        for (i, row) in df_link.iterrows():
            if int(row['F_NODE']) in G.nodes() and int(row['T_NODE']) in G.nodes():
                if int(row['LINK_ID']) in stop_links:
                    continue
                G.add_edge(
                    int(row['F_NODE']), int(row['T_NODE']),
                    link_id=int(row['LINK_ID']),
                    max_spd=row['MAX_SPD'],
                    length=row['LENGTH']
                )
        
        num_nodes = len(G.nodes())
        while True:
            node_list = list(G.nodes())
            for node in node_list:
                if len([n for n in G.neighbors(node)]) < 2:
                    G.remove_node(node)
            if len(G.nodes()) == num_nodes:
                break
            num_nodes = len(G.nodes())
        
        return G

    def set_accident(self, G, is_random=False):
        if is_random:
            # weights = (70, 18, 7, 3, 2)
            weights = (80, 12, 5, 2, 1)

            # set accident
            accident_ranks = [0, 1, 2, 3, 4]
            for edge in G.edges():
                G.edges[edge]['accident'] = random.choices(accident_ranks, weights=weights)[0]
        else:
            if self.accident_api is None:
                raise ValueError()

            def _set(link_id, rank):
                for edge in G.edges():
                    if G.edges[edge]['link_id'] == link_id:
                        G.edges[edge]['accident'] = rank
                        break

#             params = {
#                 'locationName': '세종',
#                 'numOfRows': 100,
#                 'pageNo': 1,
#                 'dataType': 'JSON',
#                 'details': 'TRUE',
#                 'startX': self.area_coords[0],
#                 'endX': self.area_coords[1],
#                 'startY': self.area_coords[2],
#                 'endY': self.area_coords[3],
#                 'vertex': 'TRUE'
#             }
#             accident_ranks = [0, 1, 2, 3, 4]
            
#             # request
#             response = requests.get(self.accident_api, params=params, timeout=10)
#             acc_link_map = response.json()
            
            import json
            accident_ranks = [0, 1, 2, 3, 4]
            with open(os.path.join(self.data_dir, 'acc_pred_example.json'), 'r') as f:
                acc_link_map = json.load(f)

            for edge in G.edges():
                G.edges[edge]['accident'] = 0

            edge_ids = [G.edges[edge]['link_id'] for edge in G.edges()]
            edges = [edge for edge in G.edges()]

            # set
            for item in acc_link_map['items']:
                if item['linkId'] in edge_ids:
                    _set(item['linkId'], item['rank'])
                    
        return G
            
    def set_speed(self, G, is_random=False):#, is_its=True):
        # set speed
        if is_random:
            for edge in G.edges():
                max_spd = G.edges[edge]['max_spd']
                _min = 1
                _max = int(max_spd*1.2)
                if _max < 10:
                    _max = 20
                spd = random.randint(_min, _max)
                G.edges[edge]['speed'] = spd
            
        else:
            def _set(link_id, speed):
                for edge in G.edges():
                    if G.edges[edge]['link_id'] == link_id:
                        G.edges[edge]['speed'] = speed
                        break

            # def now_time_set():
            #     test_datetime = datetime.datetime.now()
            #     test_datetime_kst = test_datetime.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=0)))
            #     test_datetime_utc = test_datetime_kst.astimezone(datetime.timezone.utc)
            #     nowYear = test_datetime_utc.year
            #     nowMonth = test_datetime_utc.month
            #     nowDay = test_datetime_utc.day
            #     nowHour = test_datetime_utc.hour
            #     nowMinute = test_datetime_utc.minute
            #     nowSecond = test_datetime_utc.second
            #     return '{0:04d}{1:02d}{2:02d}{3:02d}{4:02d}{5:02d}'.format(
            #         nowYear, nowMonth, nowDay, nowHour, nowMinute, nowSecond)

            def _set_none_speed():
                test_datetime = datetime.datetime.now()
                test_datetime_kst = test_datetime.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=0)))
                test_datetime_utc = test_datetime_kst.astimezone(datetime.timezone.utc)
                nowHour = test_datetime_utc.hour

                df_speed = pd.read_csv(
                    os.path.join(self.data_dir, 'AUTO_AREA_SPEED_MEAN.csv'), encoding='cp949')
                for edge in G.edges():
                    # speed 값이 있을때
                    try:
                        speed = G.edges[edge]['speed']
                        continue
                    # speed 값이 없을때
                    except:
                        is_id = df_speed['LINK_ID'] == G.edges[edge]['link_id']
                        is_hour = df_speed['HOUR'] == nowHour
                        df_idhour = df_speed[is_id & is_hour]
                        # 과거 소통정보에서 값이 없으면 max_speed로 설정
                        if len(df_idhour) == 0:
                            G.edges[edge]['speed'] = G.edges[edge]['max_spd']
                        # 과거 소통정보에 값이 있으면 과거 값을 사용
                        else:
                            G.edges[edge]['speed'] = int(df_idhour['SPEED'])
            
            # 소통정보를 api에서 받아올 때
            if self.speed_api_flag:
                apiKey = "d0b81a9264e14812bd0462e2ccecb5dd"
                url = self.speed_api

                params = {
                    'apiKey': apiKey,
                    'type': 'all',
                    'minX': self.area_coords[0],
                    'maxX': self.area_coords[1],
                    'minY': self.area_coords[2],
                    'maxY': self.area_coords[3],
                    'getType': 'json'
                }

                payload={}
                headers = {}

                response = requests.request("GET", url, params=params, headers=headers, data=payload, timeout=10)
                
                speed_link_map = response.json()

                edge_ids = [G.edges[edge]['link_id'] for edge in G.edges()]

                # _test_data = []
                for item in speed_link_map['body']['items']:
                    if int(item['linkId']) in edge_ids:
                        _set(int(item['linkId']), int(item['speed']))
                        # _test_data.append([item['linkId'], item['speed'], item['createdDate']])
            # 소통정보를 DB에서 받아올 때
            else:
                def nowTimeSet():
                    test_datetime = datetime.datetime.now()
                    test_datetime_kst = test_datetime.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=0)))
                    test_datetime_utc = test_datetime_kst.astimezone(datetime.timezone.utc)
                    nowYear = test_datetime_utc.year
                    nowMonth = test_datetime_utc.month
                    nowDay = test_datetime_utc.day
                    nowHour = test_datetime_utc.hour
                    nowMinute = test_datetime_utc.minute
                    nowSecond = test_datetime_utc.second

                    dayValue = (datetime.datetime.now() - datetime.datetime(nowYear, 1, 1, 0, 0, 0, 0)).days
                    
                    return [nowYear, nowMonth, nowDay, nowHour, nowMinute, nowSecond, dayValue]

                def connectDB_impala():
                    conn = connect(
                        host='10.100.10.71', 
                        port=21050, database='amx_cits', 
                        user='cits',
                        auth_mechanism=0
                    )
                    return conn

#                 def getLinkSpeedSetDB(conn) :
                # npLinkInfo = np.load("LINK_INFO.npy")

                liTimeSet = get.nowTimeSet()
                year = str(liTimeSet[0]).zfill(4)
                month = str(liTimeSet[1]).zfill(2)
                yyyymm = year+month

                # 최근 업데이트 req_ymd 도출
                pdData = pd.read_sql("select req_ymd from tb_bc_ht_its_curr_spd  where yyyymm='{}' ORDER BY req_ymd DESC LIMIT 1".format(yyyymm), connectDB_impala())
                req_ymd = pdData.req_ymd[0][:-2]+"%"

                # 교통소통정보 호출
                pdData = pd.read_sql("select link_id, spd from tb_bc_ht_its_curr_spd where yyyymm='{}' and req_ymd like '{}'".format(yyyymm, req_ymd), connectDB_impala())
                npData = np.array(pdData)
                npData = np.float64(npData)

                # liData = []

                edge_ids = [G.edges[edge]['link_id'] for edge in G.edges()]
                for data in npData:
                    # if data[0] in npLinkInfo :
                    if data[0] in edge_ids:
                        # liData.append(data)
                        _set(int(data[0]), int(data[1]))

            _set_none_speed()
                
            # df_speed = pd.DataFrame(_test_data)
            # df_speed.columns = ['LINK_ID', 'SPEED', 'CREATE_DATE']
            # df_speed.to_csv('./data/speed_{}.csv'.format(now_time_set()))
        return G