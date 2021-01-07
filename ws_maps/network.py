#  Copyright (c) 2020. AV Connect Inc.
"""This module provides access to road network data. """

import ws_maps.config as config
import ws_maps.models as models
import pyrosm
import networkx
import os.path
import redis
import warnings
import pandas as pd
import pyarrow as pa
import shapely.geometry
import geopandas as gpd
import osmnx
import osmnx.utils_graph
import logging
import plotly.graph_objects as go
import plotly.io as pio
from sqlalchemy import Column, String, Integer, Float, BigInteger
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, load_only
import sqlalchemy as sa
from ws_maps.pgrouting import PgrNode, PGRouting
from typing import Union, Optional, Dict, List
import re


r = config.Config().redis
engine = sa.create_engine(config.Config().gisdb.uri)
Base = declarative_base()
Base.metadata.bind = engine
db_session = sessionmaker()
db_session.bind = engine
session = scoped_session(db_session)
pio.renderers.default = "browser"
# mapbox_access_token = Config.mapbox


class Network:
    """Road network object based on OpenStreetMaps data.

    :param pbf: path to pbf file
    :type pbf: Optional[str]
    :param bbid: unique bounding box identifier in our database
    :type bbid: Optional[Union[str, Dict]]
    :param cfg: configuration object
    :type cfg: Optional[config.Config]
    :param data_backend:
    :type data_backend: Optional[str]
    :param graph_backend:
    :type graph_backend: Optional[str]
    """
    def __init__(self,
                 pbf: Optional[str] = None,
                 bbid: Optional[Union[str, Dict]] = None,
                 cfg: Optional[config.Config] = None,
                 data_backend: Optional[str] = 'postgis',
                 graph_backend: Optional[str] = 'pgrouting'):

        # validate arguments
        if cfg is not None:
            assert isinstance(cfg, config.Config), "cfg argument must be a config.Config instance"
        else:
            cfg = config.Config()

        # redis connection
        self._r = cfg.redis

        # gis db connection
        self._gisdb_engine = sa.create_engine(cfg.gisdb.uri)
        db_session = sessionmaker()
        db_session.bind = engine
        self._gisdb_session = scoped_session(db_session)

        # pgrouting db connection
        self._pgr = PGRouting(user=cfg.routingdb.user, password=cfg.routingdb.password, host=cfg.routingdb.host,
                              port=cfg.routingdb.port, database=cfg.routingdb.database)
        self._pgr.set_meta_data(**cfg.pgrouting.pop('uri'))
        self._data_backend = data_backend
        self._graph_backend = graph_backend
        self._profile = 'driving'

        if pbf is not None:
            assert isinstance(pbf, str), "pbf argument must be a string"
            assert os.path.isfile(pbf), "pbf argument must specify a valid file path"
            self._pbf = pbf
        else:
            self._pbf = None

        if bbid is not None:
            assert isinstance(bbid, str) or isinstance(bbid, dict), "bbid argument must be a string or a dictionary"
            if isinstance(bbid, str):
                self._bbid = bbid
                self._bbox = None
            else:
                assert("bbid" in bbid.keys())
                self._bbid = bbid["bbid"]
                assert ([k in bbid.keys() for k in ["min_lon", "min_lat", "max_lon", "max_lat"]])
                self._bbox = [bbid["min_lon"], bbid["min_lat"], bbid["max_lon"], bbid["max_lat"]]
        else:
            if self._pbf is not None:
                self._bbid = pbf.split('/')[-1].split('.')[0]
                self._bbox = None
            else:
                self._bbid = None

        # construct from pbf or read from db
        if self._pbf is not None:
            self._nodes, self._ways = self._parse_pbf()
            self._graph = self._make_graph(self._nodes, self._ways)
            # self._to_redis()
            self._to_gisdb()
        elif self._bbid is not None:
            self._from_redis()
            # self._from_gisdb()
        else:
            self._nodes = None
            self._ways = None
            self._edges = None
            self._graph = None

    @property
    def bbid(self):
        """

        :return:
        :rtype:
        """
        return self._bbid

    def nodes(self, ids: Optional[List[int]] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns table of OpenStreetMaps nodes in the network with attributes, indexed by id.

        :param ids: optional subset of requested ids, if omitted all available ids are returned.
        :type ids: Optional[List[int]]
        :param columns: optional subset of requested columns, if omitted all available columns are returned.
        :type columns: Optional[List[str]]
        :return: table of OpenStreetMaps nodes in the network with attributes, indexed by id.
        :rtype: gpd.GeoDataFrame
        """
        if self._nodes is not None:
            nodes = self._nodes
        else:
            if ids is None:
                if columns is None:
                    # SELECT * FROM nodes
                    sql_query = session.query(Node).statement
                else:
                    # SELECT columns FROM nodes
                    sql_query = session.query(Node).options(load_only(*columns)).statement
            else:
                if columns is None:
                    # SELECT * FROM nodes WHERE id IN ids
                    sql_query = session.query(Node).filter(Node.id.in_(ids)).statement
                else:
                    # SELECT columns FROM nodes WHERE id IN ids
                    sql_query = session.query(Node).options(load_only(*columns)).filter(Node.id.in_(ids)).statement
            nodes = pd.read_sql(sql=sql_query, con=self._gisdb)
        return nodes.set_index('id').reindex(ids)

    def ways(self, ids: Optional[List[int]] = None, columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """Returns table of OpenStreetMaps ways in the network with attributes, indexed by id.

        :param ids: optional subset of requested ids, if omitted all available ids are returned.
        :type ids: Optional[List[int]]
        :param columns: optional subset of requested columns, if omitted all available columns are returned.
        :type columns: Optional[List[str]]
        :return: table of OpenStreetMaps ways in the network with attributes, indexed by id.
        :rtype: gpd.GeoDataFrame
        """
        if self._ways is not None:
            ways = self._ways
        else:
            if ids is None:
                if columns is None:
                    # SELECT * FROM ways
                    sql_query = session.query(Way).statement
                else:
                    # SELECT columns FROM ways
                    sql_query = session.query(Way).options(load_only(*columns)).statement
            else:
                if columns is None:
                    # SELECT * FROM way WHERE id IN ids
                    sql_query = session.query(Way).filter(Way.id.in_(ids)).statement
                else:
                    # SELECT columns FROM ways WHERE id IN ids
                    sql_query = session.query(Way).options(load_only(*columns)).filter(Way.id.in_(ids)).statement
            # fixme either add 'geometry' to requested columns by default, and always return a gdf, or return df or gdf
            #  based on geometry being in the request
            ways = gpd.read_postgis(sql=sql_query, con=self._gisdb, geom_col="geometry")
        return ways.set_index('id')  # .reindex(ids)

    def edges(self, sources=None, targets=None):
        """

        :param sources:
        :type sources:
        :param targets:
        :type targets:
        :return:
        :rtype:
        """
        if self._edges is not None:
            edges = self._edges  # fixme check if source and target are not none else use loc
        else:
            if sources is None:
                if targets is None:
                    # SELECT * FROM edges
                    sql_query = session.query(Edge).statement
                    edges = pd.read_sql(sql=sql_query, con=self._gisdb)
                else:
                    # SELECT * FROM edges WHERE v IN targets
                    sql_query = session.query(Edge).filter(Edge.v.in_(targets)).statement
                    edges = pd.read_sql(sql=sql_query, con=self._gisdb)
            else:
                if targets is None:
                    # SELECT * FROM edges WHERE u IN sources
                    sql_query = session.query(Edge).filter(Edge.u.in_(sources)).statement
                    edges = pd.read_sql(sql=sql_query, con=self._gisdb)
                else:
                    # SELECT * FROM edges WHERE u IN sources AND v IN targets
                    uv_tuples = [(u, v) for u, v in zip(sources, targets)]
                    sql_query = session.query(Edge).filter(sa.tuple_(Edge.u, Edge.v).in_(uv_tuples)).statement
                    edges = pd.read_sql(sql=sql_query, con=self._gisdb)
                    # need to reorder as sqlalchemy does not support ORDER BY natively
                    edges = edges.set_index(['u', 'v']).reindex(uv_tuples).reset_index()
        # fixme id is now edges id... index by u/v?
        return edges.set_index('id').rename(columns={'u': 'source', 'v': 'target', 'way_id': 'way'})  # fixme renaming

    @property
    def graph(self, nodes=None, radius=None):
        """networkx.MultiDiGraph: routable graph of the network. """
        if nodes is None:
            return self._graph
        else:
            df_edges = []
            self._graph = networkx.convert_matrix.from_pandas_edgelist(df_edges, source='u', target='v',
                                                                       edge_attr=['id', 'key', 'length'],
                                                                       create_using=networkx.MultiDiGraph)

    def shortest_path(self, source, target, via=None, weight='length', bbox=None):
        """ Connects to pgRouting if available, otherwise uses networkx graph.

        :param source:
        :type source:
        :param target:
        :type target:
        :param via:
        :type via:
        :param weight:
        :type weight:
        """

        if self._graph_backend == 'networkx':
            return networkx.shortest_path(self._graph, source, target, weight=weight)
        else:
            if isinstance(via, list):
                return self._pgr.get_route(source, target, via_nodes=via, bbox_nodes=bbox)
            else:
                return self._pgr.get_route(source, target)

    def has_edge(self, source, target):
        """

        :param source:
        :type source:
        :param target:
        :type target:
        :return:
        :rtype:
        """
        if self._graph_backend == 'networkx':
            return self._graph.has_edge(source, target)
        else:
            return self._pgr.has_edge(source, target)

    def has_node(self, node):
        """

        :param node:
        :type node:
        :return:
        :rtype:
        """
        if self._graph_backend == 'networkx':
            return self._graph.has_node(node)
        else:
            return self._pgr.has_node(node)

    def from_pbf(self):
        pass

    def _parse_pbf(self):
        """Parses the pbf file using pyrosm and extracts nodes and ways for the selected profile. """
        # todo pyrosm bounding_box is inefficient, first extract bbox from pbf using pyosmium
        osm = pyrosm.OSM(filepath=self._pbf, bounding_box=self._bbox)
        osm.keep_node_info = True
        logging.info("Parsing OSM ways and nodes from pbf file")
        ways = osm.get_network(self._profile)

        all_nodes = osm._nodes_gdf  # todo this has all the attributes but goes out of memory easily
        way_nodes_ids = ways.nodes.explode().unique().tolist()
        nodes = all_nodes.loc[all_nodes.id.isin(way_nodes_ids)]

        def clip_way(way):
            clipped_way = way.copy()
            clipped_way.nodes = clipped_way.nodes[:len(way.geometry.coords.xy[0])-1]
            return clipped_way

        broken_ways = ways.nodes.apply(lambda x: not all(n in nodes.id.to_list() for n in x))
        for ix, way in ways.loc[broken_ways].iterrows():
            ways.loc[ix] = clip_way(way)

        # all_nodes = osm._node_coordinates
        # way_nodes = {id: all_nodes[id] for id in way_nodes_ids if id in all_nodes.keys()}
        # nodes = pd.DataFrame.from_dict(way_nodes, orient='index', columns=['longitude', 'latitude'])
        # nodes['id'] = nodes.index

        # ways_nodes_found = ways.nodes.apply(lambda x: all(n in nodes.id.to_list() for n in x))
        # nodes = pd.DataFrame(columns=['id', 'latitude', 'longitude'])
        # for ix, way in ways.iterrows():
        #     way_nodes = pd.DataFrame(columns=['id', 'latitude', 'longitude'])
        #     try:
        #         way_nodes.id = way.nodes
        #         way_nodes.latitude = way.geometry.coords.xy[1]
        #         way_nodes.longitude = way.geometry.coords.xy[0]
        #     except:
        #         pass
        #     nodes = nodes.append(way_nodes, ignore_index=True)

        logging.info("Done parsing pbf file")
        return nodes, ways

    def _to_redis(self):
        """Dumps network data to redis. """

        # OSM ways and nodes tables
        self._gdf_to_redis(self._bbid + "_ways", self._ways, geometry='geometry')
        self._df_to_redis(self._bbid + "_nodes", self._nodes)

        # graph to graph nodes and edges tables (storing only ids and edge lengths)
        gdf_nodes, gdf_edges = osmnx.utils_graph.graph_to_gdfs(self._graph, node_geometry=False,
                                                               fill_edge_geometry=False)
        self._gdf_to_redis(self._bbid + "_graph_nodes", gdf_nodes[['id']])  # ['id', 'x', 'y']  to store coordinates
        self._gdf_to_redis(self._bbid + "_graph_edges", gdf_edges[['id', 'length', 'u', 'v', 'key']])

    def _gdf_to_redis(self, key, gdf, geometry=None):
        """Dumps a gpd.GeoDataFrame to redis. """
        assert(isinstance(gdf, gpd.GeoDataFrame) or isinstance(gdf, pd.DataFrame))
        if geometry is not None and geometry in gdf.columns:
            gdf[geometry] = gdf[geometry].apply(shapely.geometry.mapping)
        self._df_to_redis(key, pd.DataFrame(gdf))

    def _df_to_redis(self, key, df):
        """Dumps a pd.DataFrame to redis. """
        assert (isinstance(df, pd.DataFrame))
        context = pa.default_serialization_context()
        self._r.set(key, context.serialize(df).to_buffer().to_pybytes())

    def _from_redis(self):
        """Loads network data from redis. """

        # OSM ways and nodes tables
        logging.info("Loading OSM ways from redis")
        self._ways = self._gdf_from_redis(self._bbid + "_ways", geometry='geometry')
        if self._ways is None:
            raise Exception("No ways data found for this bbid, please provide a pbf file or a different database")
        logging.info("Loading OSM nodes from redis")
        self._nodes = self._df_from_redis(self._bbid + "_nodes")
        if self._nodes is None:
            raise Exception("No nodes data found for this bbid, please provide a pbf file or a different database")

        # graph nodes and edges tables (storing only ids and edge lengths)
        logging.info("Loading graph nodes and edges from redis")
        df_nodes = self._df_from_redis(self._bbid + "_graph_nodes")
        df_edges = self._df_from_redis(self._bbid + "_graph_edges")
        self._edges = df_edges  # todo clean up what is stored and what not
        if df_nodes is None or df_edges is None:
            warnings.warn("Found ways and nodes data, but not a routing graph for this bbid. Making graph now.")
            self._make_graph(self._nodes, self._ways)
            gdf_nodes, gdf_edges = osmnx.utils_graph.graph_to_gdfs(self._graph, node_geometry=False,
                                                                   fill_edge_geometry=False)
            self._gdf_to_redis(self._bbid + "_graph_nodes", gdf_nodes[['id']])  # ['id', 'x', 'y']  to store coordinates
            self._gdf_to_redis(self._bbid + "_graph_edges", gdf_edges[['id', 'length', 'u', 'v', 'key']])
        else:
            logging.info("Reassembling graph")
            self._graph = networkx.convert_matrix.from_pandas_edgelist(df_edges, source='u', target='v',
                                                                       edge_attr=['id', 'key', 'length'],
                                                                       create_using=networkx.MultiDiGraph)
        logging.info("Done loading from redis")

    def _gdf_from_redis(self, key, geometry=None):
        """Loads a gpd.GeoDataFrame from redis. """
        df = self._df_from_redis(key)
        if df is not None:
            gdf = gpd.GeoDataFrame(df)
            if geometry is not None and geometry in gdf.columns:
                gdf[geometry] = gdf[geometry].apply(shapely.geometry.shape)
            return gdf
        else:
            return None

    def _df_from_redis(self, key):
        """Loads a pd.DataFrame from redis. """
        buffer = self._r.get(key)
        if buffer is not None:
            context = pa.default_serialization_context()
            return context.deserialize(buffer)
        else:
            return None

    def _to_gisdb(self):
        """Dumps network data to GIS database. """
        self._ways.to_postgis(name="ways", con=self._gisdb, if_exists="append")
        self._nodes.to_sql(name="nodes", con=self._gisdb, if_exists="append")
        gdf_nodes, gdf_edges = osmnx.utils_graph.graph_to_gdfs(self._graph, node_geometry=False,
                                                               fill_edge_geometry=False)
        gdf_edges[['id', 'length', 'u', 'v', 'key']].to_postgis(name="graph_edges", con=self._gisdb, if_exists="append")
        gdf_nodes[['id']].to_postgis(name="graph_nodes", con=self._gisdb, if_exists="append")
        self._nodes.to_sql(name="nodes", con=self._gisdb, if_exists="append")

    def _from_gisdb(self):
        """Loads network data from GIS database. """
        self._ways = gpd.read_postgis(sql="ways", con=self._gisdb, geom_col="geometry")
        self._nodes = pd.read_sql(sql="nodes", con=self._gisdb)
        self._edges = pd.read_sql(sql="graph_edges", con=self._gisdb)
        # graph_nodes = gpd.read_postgis(sql="graph_nodes", con=self._gisdb, geom_col="geometry")

    @staticmethod
    def _make_graph(nodes, ways):
        """Makes a routable networkx.MultiDiGraph out of the network ways data. """
        graph = networkx.MultiDiGraph(crs="EPSG:4326")
        ways_proj = ways.set_crs("EPSG:4326").to_crs("EPSG:3395")

        for node_id, node_attr in nodes.rename(columns={'longitude': 'x', 'latitude': 'y'}).iterrows():
            graph.add_node(node_id, **node_attr)

        for _, way in ways_proj.iterrows():

            osm_oneway_values = ["yes", "true", "1", "-1", "T", "F"]
            if "oneway" in way and way.oneway in osm_oneway_values:
                if way["oneway"] == "-1" or way["oneway"] == "T":
                    # paths with a one-way value of -1 or T are one-way, but in the
                    # reverse direction of the nodes' order, see osm documentation
                    path_nodes = list(reversed(way.nodes))
                else:
                    path_nodes = way.nodes
                # add this path (in only one direction) to the graph
                one_way = True

            elif "junction" in way and way.junction == "roundabout":
                # roundabout are also oneway but not tagged as is
                path_nodes = way.nodes
                one_way = True

                # else, this path is not tagged as one-way or it is a walking network
                # (you can walk both directions on a one-way street)
            else:
                # add this path (in both directions) to the graph and set its
                # 'oneway' attribute to False. if this is a walking network, this
                # may very well be a one-way street (as cars/bikes go), but in a
                # walking-only network it is a bi-directional edge
                path_nodes = way.nodes
                one_way = False

            # zip together the path nodes so you get tuples like (0,1), (1,2), (2,3)
            # and so on
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            graph.add_edges_from(path_edges, **way[['id']])
            if not one_way:
                path_edges_reverse = [(v, u) for u, v in path_edges]
                graph.add_edges_from(path_edges_reverse, **way[['id']])

        graph = osmnx.utils_graph.add_edge_lengths(graph)
        return graph

    def plot(self):
        """Plots the network. """
        return self._plot_plotly()

    def _add_edge(self, start_lon, start_lat, end_lon, end_lat, edge_lon, edge_lat):

        # Append line corresponding to the edge
        edge_lon.append(start_lon)
        edge_lon.append(end_lon)
        edge_lon.append(None)  # Prevents a line being drawn from end of this edge to start of next edge
        edge_lat.append(start_lat)
        edge_lat.append(end_lat)
        edge_lat.append(None)

        return edge_lon, edge_lat

    def _plot_plotly(self):
        edges_with_props = self._edges.copy()
        edges_with_props['u_latitude'] = self._nodes.loc[self._edges.u.values, 'latitude'].values
        edges_with_props['u_longitude'] = self._nodes.loc[self._edges.u.values, 'longitude'].values
        edges_with_props['v_latitude'] = self._nodes.loc[self._edges.v.values, 'latitude'].values
        edges_with_props['v_longitude'] = self._nodes.loc[self._edges.v.values, 'longitude'].values

        edges_lon = []
        edges_lat = []
        for ix, edge in edges_with_props.iterrows():
            edges_lon, edges_lat = self._add_edge(
                edge.u_longitude, edge.u_latitude, edge.v_longitude, edge.v_latitude, edges_lon, edges_lat)

        fig = go.Figure()
        fig.add_scattermapbox(lat=self._nodes.latitude, lon=self._nodes.longitude, mode='markers', name='nodes')
        fig.add_scattermapbox(lat=edges_lat, lon=edges_lon, mode='lines', name='edges')
        # todo color by node and way types
        # todo add node/way properties to annotations
        # todo bidirectional edges render properly
        fig.update_layout(
            showlegend=True,
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(
                    lat=self._nodes.latitude.mean(),
                    lon=self._nodes.longitude.mean()
                )
            )
        )
        fig.show()
        return fig

    def highlight_edges(self, fig, edges):
        """

        :param fig:
        :type fig:
        :param edges:
        :type edges:
        """
        edges_with_props = edges.copy()
        edges_with_props['u_latitude'] = self._nodes.loc[edges.source.values, 'latitude'].values
        edges_with_props['u_longitude'] = self._nodes.loc[edges.source.values, 'longitude'].values
        edges_with_props['v_latitude'] = self._nodes.loc[edges.target.values, 'latitude'].values
        edges_with_props['v_longitude'] = self._nodes.loc[edges.target.values, 'longitude'].values

        edges_lon = []
        edges_lat = []
        for ix, edge in edges_with_props.iterrows():
            edges_lon, edges_lat = self._add_edge(
                edge.u_longitude, edge.u_latitude, edge.v_longitude, edge.v_latitude, edges_lon, edges_lat)

        fig.add_scattermapbox(lat=edges_lat, lon=edges_lon, mode='lines', name='edges',
                              line=dict(width=5, color='orange'))
        fig.show()


class BBID(models.Base, models.ConnectedModel):
    """Models a rectangular geographic area defined by a name and its minimum and maximum latitude and longitude. """
    __tablename__ = 'bbid'
    id = Column(Integer, primary_key=True)
    bbid = Column(String(64), index=True)
    max_lat = Column(Float, nullable=True)
    min_lat = Column(Float, nullable=True)
    max_lon = Column(Float, nullable=True)
    min_lon = Column(Float, nullable=True)

    def __str__(self):
        return self.bbid

    def contains_point(self, lat: float, lon: float) -> bool:
        """Returns true if the provided coordinates are within the bounding box.

        :param lat: the point's latitude.
        :type lat: float
        :param lon: the point's longitude.
        :type lon: float
        :return: true if the provided coordinates are within the bounding box.
        :rtype: bool
        """
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def center_point(self) -> tuple:
        """Returns the center point of the bounding box.

        :return: latitude and longitude of the center point.
        :rtype: tuple
        """
        return (self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2

    def get_network(self, red=r):
        """Returns a network object for this bbid.

        :param red: optional redis connection
        :type red: redis.Redis
        :return: network object for this bbid
        :rtype: Network
        """
        bbid_dict = {'bbid': self.bbid,
                     'min_lat': self.min_lat,
                     'max_lat': self.max_lat,
                     'min_lon': self.min_lon,
                     'max_lon': self.max_lon
                     }
        return Network(bbid=bbid_dict, r=red)


class Node(Base):
    """Models an OpenStreetMap node. """

    __tablename__ = 'nodes'
    id = Column(BigInteger, primary_key=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    # todo tags?


class Way(Base):
    """Models an OpenStreetMap way. """

    __tablename__ = 'ways'
    id = Column(BigInteger, primary_key=True)
    access = Column(String, nullable=True)
    area = Column(String, nullable=True)
    bicycle = Column(String, nullable=True)
    bridge = Column(String, nullable=True)
    busway = Column(String, nullable=True)
    cycleway = Column(String, nullable=True)
    est_width = Column(String, nullable=True)
    foot = Column(String, nullable=True)
    footway = Column(String, nullable=True)
    highway = Column(String, nullable=True)
    junction = Column(String, nullable=True)
    lanes = Column(String, nullable=True)
    lit = Column(String, nullable=True)
    maxspeed = Column(String, nullable=True)
    motorcar = Column(String, nullable=True)
    # motorroad = Column(Float, nullable=True)
    motor_vehicle = Column(String, nullable=True)
    name = Column(String, nullable=True)
    oneway = Column(String, nullable=True)
    overtaking = Column(String, nullable=True)
    # passing_places = Column(Float, nullable=True)
    psv = Column(String, nullable=True)
    ref = Column(String, nullable=True)
    service = Column(String, nullable=True)
    segregated = Column(String, nullable=True)
    sidewalk = Column(String, nullable=True)
    smoothness = Column(String, nullable=True)
    surface = Column(String, nullable=True)
    tracktype = Column(String, nullable=True)
    tunnel = Column(String, nullable=True)
    turn = Column(String, nullable=True)
    width = Column(String, nullable=True)
    nodes = Column(String, nullable=True)
    timestamp = Column(BigInteger, nullable=True)
    version = Column(Integer, nullable=True)
    tags = Column(String, nullable=True)
    geometry = Column(Geometry, nullable=True)
    osm_type = Column(String, nullable=True)


class Edge(Base):
    """Models an OpenStreetMap segment defined as the line connecting two nodes in a way. """

    __tablename__ = 'edges'
    id = Column(BigInteger, primary_key=True)
    way_id = Column(BigInteger, nullable=False)
    u = Column(BigInteger, nullable=False, index=True)  # source
    v = Column(BigInteger, nullable=False, index=True)  # target
    length = Column(Float, nullable=True)

