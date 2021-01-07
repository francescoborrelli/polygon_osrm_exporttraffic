#  Copyright (c) 2020. AV Connect Inc.
"""This module provides an interface between road network data and TMC codes used in traffic data.
https://pdfs.semanticscholar.org/a263/2aa92472adfec9c273bde9b8657f561b5826.pdf
https://gis.stackexchange.com/questions/309164/matching-streets-in-two-line-datasets-with-postgis-or-qgis
http://postgis.net/docs/ST_Snap.html
https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
https://gis.stackexchange.com/questions/202895/snapping-two-lines-using-shapely
https://gis.stackexchange.com/questions/260495/snap-lines-to-nearest-line-in-postgis
https://geopandas.org/projections.html
https://gis.stackexchange.com/questions/203058/why-is-shapelys-snapping-geo-snaps-not-working-as-expected
"""

import ws_maps.config as config
import ws_maps.network as network
import ws_maps.match as match
import ws_maps.osrm as osrm
import ws_maps.here as here
from shapely.geometry import LineString
import shapely.ops
import shapely.wkt
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx
import plotly.graph_objects as go
import pyarrow as pa
import time
from sqlalchemy import Column, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import sqlalchemy as sa

engine = sa.create_engine('postgresql://postgres:postgres@localhost:5432/gisdb')
Base = declarative_base()
Base.metadata.bind = engine
db_session = sessionmaker()
db_session.bind = engine
session = scoped_session(db_session)


class Tmc:
    """This class provides methods to create and update a table of TMC codes and their corresponding OSM entities

    The class is initialized with a WS bbid (which identifies a subset of the planet's OpenStreetMap data) and a traffic
    blob (from Here's Traffic Flow API). The traffic blob contains traffic flow information at different locations in
    the road network. Every traffic flow item is associated with a shapefile (a set of lat/lon coordinates) and a unique
    TMC code. The class extracts the TMC codes along with their shapefiles, performs map matching to define a mapping
    between TMC codes and WS/OSM nodes, and persists the results in the WS database. When an instance is created, the
    TMC codes in the traffic blob are searched among the known ones, so the matching is only performed when the traffic
    blob is for a new geographic area or it contains new codes due to map updates.

    :param bbid: unique bounding box identifier in WS database
    :param traffic_blob: traffic blob from Here's Traffic Flow API
    :type bbid:
    :type traffic_blob: dict
    """
    KPH2MPS = 1.0 / 3.6

    def __init__(self, bbid, traffic_blob):
        self._network = None
        self._bbid = bbid
        self._way_ids = None
        self._lengths = None
        self._config = config.Config()

        # look up any existing tmc table for this bbid; todo if rows are missing, add them
        ws_to_tmc_redis = self._df_from_redis(self._bbid + '_ws_to_tmc')
        tmc_to_ws_redis = self._df_from_redis(self._bbid + '_tmc_to_ws')
        self._gdf = here.TrafficFlow.json_to_gdf(traffic_blob)
        if ws_to_tmc_redis is not None and tmc_to_ws_redis is not None:
            self._ws_to_tmc = ws_to_tmc_redis
            self._tmc_to_ws = tmc_to_ws_redis
        else:
            self._network = network.Network(bbid=bbid)
            # self._way_ids = networkx.get_edge_attributes(self._network.graph, 'id')
            # self._lengths = networkx.get_edge_attributes(self._network.graph, 'length')
            edges = self._network.edges(sources=[], targets=[])  # fixme get from response source/targets
            self._way_ids = edges.id
            self._lengths = edges.length
            self._ws_to_tmc, self._tmc_to_ws = self._match_table(self._gdf)
            self._df_to_redis(self._bbid + '_ws_to_tmc', self._ws_to_tmc)
            self._df_to_redis(self._bbid + '_tmc_to_ws', self._tmc_to_ws)
            # todo persist in postgres

    def traffic(self, edges):
        """

        :param edges:
        :return:
        """
        edges_traffic = pd.DataFrame(columns=['SP', 'SU', 'FF', 'JF', 'CN'], index=edges.index)
        # SP': fi['CF'][0]['SP'] if 'SP' in fi['CF'][0].keys() else None,
        #                             'SU': fi['CF'][0]['SU'] if 'SU' in fi['CF'][0].keys() else None,
        #                             'FF': fi['CF'][0]['FF'] if 'FF' in fi['CF'][0].keys() else None,
        #                             'JF
        df = pd.DataFrame(self._gdf.copy())
        df['code'] = df.apply(lambda x: str(x.PC) + x.QD, axis=1)
        df = df.set_index('code')
        self._ws_to_tmc = self._ws_to_tmc.set_index(['source', 'target'])

        # fixme this is both slow and does not check that edges actually uses source/target as index
        for index, _ in edges.iterrows():
            try:
                edges_traffic.loc[index] = df.loc[self._ws_to_tmc.loc[index, 'code'],
                                                  ['SP', 'SU', 'FF', 'JF', 'CN']].mean(axis=0)
            except KeyError:
                continue
            except ValueError:
                continue

        self._ws_to_tmc = self._ws_to_tmc.reset_index()

        edges_traffic.rename(columns={'SP': 'traffic_speed_capped', 'SU': 'traffic_speed_uncapped',
                                      'FF': 'free_flow_speed', 'JF': 'jam_factor', 'CN': 'confidence'},
                             inplace=True)
        edges_traffic['traffic_speed_capped'] = edges_traffic['traffic_speed_capped'] * self.KPH2MPS
        edges_traffic['traffic_speed_uncapped'] = edges_traffic['traffic_speed_uncapped'] * self.KPH2MPS
        edges_traffic['free_flow_speed'] = edges_traffic['free_flow_speed'] * self.KPH2MPS
        if 'order' in edges.columns:
            edges_traffic['order'] = edges.order
        return edges_traffic

    def _match_table(self, gdf):
        """Performs map matching on all the traffic flow items (rows) in the table

        :param gdf: traffic flow items table
        :type gdf: gpd.GeoDataFrame
        :return:
        :rtype:
        """
        results = {}
        # for ix, group in gdf.groupby(['DE_RW', 'QD']):
        n_groups = len(gdf.groupby('RW'))
        for ix, group in gdf.groupby('RW'):
            results[ix] = self._match_roadway(group)
            print("matched roadway {} of {}".format(ix, n_groups))
        print("ws to tmc table")
        ws_to_tmc = self._make_ws_to_tmc(results)
        print("tmc to ws table")
        tmc_to_ws = self._make_tmc_to_ws(ws_to_tmc)
        print("done")
        return ws_to_tmc, tmc_to_ws

    def _match_roadway(self, road):
        """Performs map matching on a set of traffic flow items on the same roadway

        :param road: group of traffic flow items with the same roadway identifier
        :type road: gpd.GeoDataFrame
        :return: matching results
        :rtype: dict
        """

        road['coord_list'] = road.apply(lambda x: [{'longitude': y[1],
                                                    'latitude': y[0],
                                                    'RWS': x['RWS'],
                                                    'RW': x['RW'],
                                                    'DE_RW': x['DE_RW'],
                                                    'LI': x['LI'],
                                                    'FIS': x['FIS'],
                                                    'FI': x['FI'],
                                                    'PC': x['PC'],
                                                    'LE': x['LE'],
                                                    'DE': x['DE'],
                                                    'QD': x['QD']
                                                    } for y in x['geometry'].coords], axis=1)
        trace = []
        for segment in road['coord_list'].to_list():
            subtrace = [point for point in segment]
            trace += subtrace

        trace = pd.DataFrame(trace)
        osrm_match = osrm.Match(trace=trace, use_waypoints=True)
        if osrm_match.matches is not None:
            route = match.Match._matchings_to_route(self._network,
                                                    osrm_match.matches.matchings,
                                                    way_ids=self._way_ids,
                                                    lengths=self._lengths)
            trace['datetime'] = None
            trajectory = match.Match._tracepoints_to_trajectory(trace,
                                                                osrm_match.matches.tracepoints,
                                                                route,
                                                                osrm_match.trace_index)
            # fixme up to here same as match.Match - just better to skip the filtered trace and the arcs parts for
            #  efficiency
            # todo do this with a join/merge?
            matched_trace = trace.copy()
            matched_trace['matched_longitude'] = trajectory['matched_longitude']
            matched_trace['matched_latitude'] = trajectory['matched_latitude']
            matched_trace['matching_index'] = trajectory['matching_index']
            matched_trace['waypoint_index'] = trajectory['waypoint_index']
            matched_trace['edge_index'] = trajectory['edge_index']
            matched_trace['matched_edge'] = trajectory['matched_edge']
            matched_trace['matched_source'] = trajectory['matched_source']
            matched_trace['matched_target'] = trajectory['matched_target']
            matched_trace['codes'] = matched_trace.apply(lambda x: str(x.PC) + x.QD, axis=1)

            # todo compute weights based on edge lengths vs distance covered; 100% for a fully covered edge

            def get_codes(x):
                matched_trace_rows = matched_trace.loc[(matched_trace.matched_source == x.source) &
                                                       (matched_trace.matched_target == x.target)]
                codes = np.unique(matched_trace_rows['codes'].values)
                if len(codes) > 0:
                    return codes
                else:
                    return None

            matched_nodes = route['df'].copy()
            try:
                matched_nodes['codes'] = matched_nodes.apply(get_codes, axis=1).bfill().ffill()
            except:  # fixme
                matched_nodes['codes'] = None

            return {'trace': matched_trace, 'route': matched_nodes}

    def _make_ws_to_tmc(self, match_results):
        """Creates the mapping from WS edges to TMC codes

        :param match_results:
        :type match_results: dict
        :return: table
        :rtype: pd.DataFrame
        """

        def expand_table(x):
            if x.codes is None:  # fixme this is due to line 157
                return pd.DataFrame(columns=['source', 'target', 'RW', 'code'])
            elif len(x.codes) <= 1:
                s = x[['source', 'target', 'RW']]  # 'name', 'direction']]
                s['code'] = x['codes'][0]
                return s.to_frame().T.reset_index(drop=True)
            else:
                s = pd.DataFrame({
                    'source': [x.source] * len(x.codes),
                    'target': [x.target] * len(x.codes),
                    'RW': [x.RW] * len(x.codes),
                    # 'name': x.name * len(x.codes),
                    # 'direction': x.direction * len(x.codes),
                    'code': x.codes.flatten().tolist()
                })
                return s

        ws_to_tmc = None
        for key, val in match_results.items():
            df = val['route'][['source', 'target', 'codes']].reset_index(drop=True)
            df['RW'] = key
            # df['name'] = key[0]
            # df['direction'] = key[1]
            for _, row in df.iterrows():
                if ws_to_tmc is None:
                    ws_to_tmc = expand_table(row)
                else:
                    ws_to_tmc = ws_to_tmc.append(expand_table(row))

        return ws_to_tmc

    def _make_tmc_to_ws(self, ws_to_tmc):
        """Creates the mapping from TMC codes to WS edges

        :param ws_to_tmc:
        :type ws_to_tmc: pd.DataFrame
        :return:
        :rtype: pd.DataFrame
        """
        def get_code_row(x):
            return pd.DataFrame({
                'edges': [(row.source, row.target) for _, row in x[['source', 'target']].iterrows()],
                'RW': x.RW.values[0]
            })

        tmc_to_ws = ws_to_tmc.groupby('code').apply(get_code_row)
        return tmc_to_ws

    def _df_to_redis(self, key, df):
        assert (isinstance(df, pd.DataFrame))
        context = pa.default_serialization_context()
        self._config.redis.set(key, context.serialize(df).to_buffer().to_pybytes())

    def _df_from_redis(self, key):
        buffer = self._config.redis.get(key)
        if buffer is not None:
            context = pa.default_serialization_context()
            return context.deserialize(buffer)
        else:
            return None

    def plot(self):
        """

        :return: plotly figure handle
        :rtype: go.Figure
        """
        if self._network is None:
            NotImplementedError("network info was not loaded")
            return None
        else:
            fig = go.Figure()
            fig = self._plot_tmc(fig)
            fig = self._plot_ws(fig)
            fig.update_layout(
                showlegend=True,
                mapbox=dict(
                    style='open-street-map',
                    zoom=10,
                    center=dict(
                        lat=np.nanmean(fig.data[0].lat),
                        lon=np.nanmean(fig.data[0].lon)
                    )
                )
            )
            fig.show()
            return fig

    def _plot_tmc(self, fig):
        """

        :param fig:
        :type fig: go.Figure
        :return:
        :rtype: go.Figure
        """
        lat = []
        lon = []
        speed = []
        for ix, row in self._gdf.iterrows():
            coords = row.geometry.coords.xy
            lat += coords[0].tolist()
            lon += coords[1].tolist()
            speed += [row.SP] * len(coords[0])
        fig.add_scattermapbox(
            lat=lat,
            lon=lon,
            mode='markers',
            name='tmc',
            marker=dict(
                opacity=0.5,
                size=10,
                color=speed,
                cmin=0,
                cmax=120,
                colorscale="thermal",
                showscale=True
            )
        )
        return fig

    def _plot_ws(self, fig):
        """

        :param fig:
        :type fig: go.Figure
        :return:
        :rtype: go.Figure
        """

        # look up latitude and longitude of edges sources and targets
        table = self._ws_to_tmc.copy()
        table['source_latitude'] = self._network.nodes(ids=table.source.values, columns=['latitude']).values
        table['source_longitude'] = self._network.nodes(ids=table.source.values, columns=['longitude']).values
        table['target_latitude'] = self._network.nodes(ids=table.target.values, columns=['latitude']).values
        table['target_longitude'] = self._network.nodes(ids=table.target.values, columns=['longitude']).values

        # look up edges matched traffic
        gdf = self._gdf.copy()
        gdf['code'] = gdf.apply(lambda x: str(x.PC) + x.QD, axis=1)
        gdf.set_index('code', inplace=True)
        table['CN'] = gdf.loc[table.code, 'CN'].values
        table['TY'] = gdf.loc[table.code, 'TY'].values
        table['SP'] = gdf.loc[table.code, 'SP'].values
        table['SU'] = gdf.loc[table.code, 'SU'].values
        table['FF'] = gdf.loc[table.code, 'FF'].values
        table['JF'] = gdf.loc[table.code, 'JF'].values

        # put data into lists for simplified plotting
        edges_lon = []
        edges_lat = []
        edges_cn = []
        edges_ty = []
        edges_sp = []
        edges_su = []
        edges_ff = []
        edges_jf = []
        for ix, edge in table.iterrows():
            edges_lon += [edge.source_longitude, edge.target_longitude]
            edges_lat += [edge.source_latitude, edge.target_latitude]
            edges_cn += [edge.CN, edge.CN]
            edges_ty += [edge.TY, edge.TY]
            edges_sp += [edge.SP, edge.SP]
            edges_su += [edge.SU, edge.SU]
            edges_ff += [edge.FF, edge.FF]
            edges_jf += [edge.JF, edge.JF]

        # add to plotly figure
        fig.add_scattermapbox(
            lat=edges_lat,
            lon=edges_lon,
            mode='markers',
            name='ws',
            marker=dict(
                opacity=0.5,
                size=10,
                cmin=0,
                cmax=120,
                color=edges_sp,
                colorscale="thermal",
                showscale=True
            ))

        return fig


class Ws2Tmc(Base):

    __tablename__ = 'ws_to_tmc'
    id = Column(BigInteger, primary_key=True)
    source = Column(BigInteger, nullable=False, index=True)
    target = Column(BigInteger, nullable=False, index=True)
    RW = Column(BigInteger, nullable=False)
    code = Column(String, nullable=True)


class Tmc2Ws(Base):

    __tablename__ = 'tmc_to_ws'
    id = Column(BigInteger, primary_key=True)
    edges = Column(BigInteger, nullable=False)
    RW = Column(BigInteger, nullable=False)

