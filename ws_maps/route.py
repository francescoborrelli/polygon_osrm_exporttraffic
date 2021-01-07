#  Copyright (c) 2020. AV Connect Inc.
"""This module provides routing functionality. """

import requests
import warnings
import pandas as pd
import shapely
import shapely.wkt
from ws_maps.config import Config
from ws_maps.network import Network, BBID
import ws_maps.models as models
import ws_maps.osrm as osrm
import networkx as nx
import plotly.graph_objects as go
from more_itertools import pairwise
from itertools import groupby
from sqlalchemy import Column, DateTime, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship
from ws_maps.traffic import Traffic
import polyline
from ws_maps.pgrouting import PgrNode


class Waypoint:
    """A location visited by a route, including origin, final and intermediate destinations.

    :param latitude:
    :type latitude: float
    :param longitude:
    :type longitude: float
    :param id:
    :type id:
    :param address:
    :type address: str
    :param date_time:
    :type date_time:
    """
    def __init__(self, latitude: float = None, longitude: float = None, id=None, address: str = None, date_time=None):

        self._config = Config()

        assert ((latitude is not None) & (longitude is not None)) ^ (id is not None) ^ (address is not None)

        if latitude is not None:
            self._latitude = latitude
            self._longitude = longitude

        if id is not None:
            self._latitude, self._longitude = self._lookup_id(id)

        if address is not None:
            self._latitude, self._longitude = self._lookup_address(address)

        self._datetime = date_time

    def _lookup_id(self, id):
        return None, None

    def _lookup_address(self, address):
        params = {
            'app_id': self._config.here['app_id'],
            'app_code': self._config.here['app_code'],
            'searchtext': address
        }
        url = 'https://geocoder.api.here.com/6.2/geocode.json'
        response = requests.get(url, params)
        # todo check response
        json_response = response.json()
        coordinates = json_response['Response']['View'][0]['Result'][0]['Location']['NavigationPosition'][0]
        return coordinates['Latitude'], coordinates['Longitude']

    @property
    def latitude(self):
        """

        :return:
        :rtype:
        """
        return self._latitude

    @property
    def longitude(self):
        """

        :return:
        :rtype:
        """
        return self._longitude

    @property
    def datetime(self):
        """

        :return:
        :rtype:
        """
        return self._datetime


class Origin(Waypoint):
    """The location at the beginning of a route.

    """
    pass


class Destination(Waypoint):
    """The location at the end of a route.

    """
    pass


@pd.api.extensions.register_dataframe_accessor("path")
class PathAccessor:
    """

    :param pandas_obj:
    :type pandas_obj: pd.DataFrame
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if 'source' not in obj.columns or 'target' not in obj.columns:
            raise AttributeError("Must have 'source' and 'target'.")

    @property
    def nodes(self) -> list:
        """

        :return:
        :rtype: list
        """
        # # node path: given source = [a, b, c, b] and target = [b, c, b, d], node path is ...
        # #  [a, b, b, c, c, b, b, d]
        # node_path = pd.concat([self._obj['source'], self._obj['target']]).sort_index().reset_index(drop=True)
        # #  [a, b, c, b, d]
        # node_path = node_path.loc[node_path != node_path.shift(-1)]
        # return node_path.tolist()
        return self._obj.source.tolist() + [self._obj.target.tolist()[-1]]

    @property
    def ways(self) -> list:
        """

        :return:
        :rtype: list
        """
        return self._obj['way'].tolist()

    @property
    def edges(self) -> pd.DataFrame:
        """

        :return:
        :rtype: pd.DataFrame
        """
        return self._obj.set_index(['source', 'target'])[['way', 'length']]


class RouteClient:
    """Models a path on a road network graph, as a sequence of adjacent edges (nodes).

    :param from_networkx:
    :type from_networkx:
    :param from_here:
    :type from_here:
    :param from_osrm:
    :type from_osrm:
    :param bbid:
    :type bbid:
    :param network:
    :type network:
    :param terrain:
    :type terrain:
    :param traffic:
    :type traffic:
    :param weather:
    :type weather:
    """
    def __init__(self, from_networkx=None, from_here=None, from_osrm=None, bbid=None, network=None,
                 terrain: bool = True, traffic: bool = False, weather: bool = True, matched: bool = False):

        self._matched = matched

        if network is None and isinstance(bbid, str):
            self._network = Network(bbid=bbid)
            self._bbid = bbid
        elif isinstance(network, Network):
            self._network = network
            self._bbid = network.bbid
        else:
            raise Exception("either network or bbid must be valid")

        assert (from_networkx is not None) ^ (from_osrm is not None) ^ (from_here is not None)

        self._here_dict = None
        if from_networkx is not None:
            # self._path = self._from_networkx(from_networkx)
            self._path = self._from_pgrouting(from_networkx)
        if from_osrm is not None:
            self._path = self._from_osrm(from_osrm)
        if from_here is not None:
            self._path = self._from_here(from_here)

        if self._matched:
            self._nodes = self._network.nodes(ids=self._path.path.nodes)
            self._ways = self._network.ways(ids=self._path.path.ways)
            mph2mps = 0.44704
            self._ways['speed_limit'] = self._ways.maxspeed.apply(
                lambda x: float(x.split(" ")[0]) * mph2mps if x is not None else x).values
            self._edges = self._path.path.edges
            try:
                self._edges['speed_limit'] = self._ways.loc[self._edges.way, 'speed_limit'].values
            except KeyError:
                self._edges['speed_limit'] = None
            del self._network
        else:
            self._edges = self._path.rename(columns={'trafficSpeed': 'traffic_speed', 'baseSpeed': 'base_speed',
                                                     'trafficTime': 'traffic_duration', 'baseTime': 'base_duration',
                                                     'jamFactor': 'jam_factor', 'speedLimit': 'speed_limit'})
            self._edges['source'] = range(0, len(self._edges))
            self._edges['target'] = self._edges['source'] + 1
            if from_here is not None:
                self._nodes = pd.DataFrame(self._path['shape'].apply(
                    lambda x: {'longitude': x.coords[0][0], 'latitude': x.coords[0][1]}).tolist())
                self._nodes = self._nodes.append({'longitude': self._path.iloc[-1]['shape'].coords[-1][0],
                                                  'latitude': self._path.iloc[-1]['shape'].coords[-1][1]},
                                                 ignore_index=True)
            elif from_osrm is not None:
                self._nodes = self._path[['source_longitude', 'source_latitude']].copy().rename(
                    columns={'source_longitude': 'longitude', 'source_latitude': 'latitude'})
                self._nodes = self._nodes.append(self._path[['target_longitude', 'target_latitude']].copy().rename(
                    columns={'target_longitude': 'longitude', 'target_latitude': 'latitude'}).iloc[-1],
                                                 ignore_index=True)
                self._nodes = self._nodes.reset_index()
            self._nodes.index.name = 'id'
            self._ways = None

        if terrain:
            from ws_maps.terrain import Terrain
            self._nodes, self._edges = Terrain().at(locations=self)

        if weather:
            from ws_maps.weather import Weather
            self._nodes, self._edges = Weather().at(locations=self)

        if traffic:
            from ws_maps.traffic import TrafficClient
            if self._bbid is None:
                min_lon = self._nodes.longitude.min()
                max_lon = self._nodes.longitude.max()
                min_lat = self._nodes.latitude.min()
                max_lat = self._nodes.latitude.max()
                for bbid in BBID.all():
                    if (bbid.contains_point(min_lat, min_lon) and bbid.contains_point(min_lat, max_lon) and
                            bbid.contains_point(max_lat, min_lon) and bbid.contains_point(max_lat, max_lon)):
                        self._bbid = bbid.bbid
            self._edges = TrafficClient().at(locations=self)
            self._edges['traffic_speed'] = self._edges.traffic_speed_uncapped.fillna(self._edges.speed_limit)

    @property
    def nodes(self):
        """

        :return:
        :rtype:
        """
        return self._nodes

    @property
    def edges(self):
        """

        :return:
        :rtype:
        """
        return self._edges

    @property
    def ways(self):
        """

        :return:
        :rtype:
        """
        return self._ways

    @property
    def path(self):
        """

        :return:
        :rtype:
        """
        return self._path

    @property
    def bbid(self):
        """

        :return:
        :rtype:
        """
        return self._bbid

    def _from_networkx(self, nodes):
        # remove any consecutive duplicates (can happen when nodes is a path from OSRM)
        visited_nodes = [i[0] for i in groupby(nodes) if self._network.has_node(i[0])]

        # convert the visited_nodes to and actual node_path, by making sure all node pairs have a connecting edge; if
        # they don't, find the shortest path connecting them (this again can happen when nodes is a path from OSRM and
        # there's a mismatch in the network data between ws and osrm
        node_path = [visited_nodes[0]]
        for s, t in pairwise(visited_nodes):
            if self._network.has_edge(s, t):
                node_path.append(t)
            else:
                warnings.warn("No edge from {} to {} was found, searching a path.".format(s, t))
                path_reconnection = self._network.shortest_path(s, t, weight='length')
                node_path += path_reconnection[1:]
        path = pd.DataFrame()
        path['source'] = node_path[:-1]
        path['target'] = node_path[1:]
        edges = self._network.edges.set_index(['u', 'v']).loc[
            pd.MultiIndex.from_frame(path[['source', 'target']], names=['u', 'v'])]
        path = self._network.edges(sources=node_path[:-1], targets=node_path[1:]).reset_index(drop=True)
        path['way'] = edges['id'].values
        path['length'] = edges['length'].values
        return path

    def _from_pgrouting(self, nodes):
        unique_nodes = [i[0] for i in groupby(nodes)]
        path = self._network.shortest_path(unique_nodes[0], unique_nodes[-1], via=unique_nodes[1:-1], weight='length')
        return path

    def _from_osrm(self, osrm_route):
        """Converts an OSRM Route object to a corresponding WideSense route

        :return:
        """
        route = pd.DataFrame()

        if self._matched:
            for lix, leg in enumerate(osrm_route.legs):
                # node_path = leg.annotation.nodes
                # # todo In some instances OSRM returns the same node repeated twice, the corresponding annotation distance
                # #  equal to 0 and the corresponding annotation duration non 0; possibly a bug in the creation of the
                # #  annotation. If OSRM and WS are run on different versions of the OSM data, it is possible that OSRM will
                # #  return nodes that don't exist in WS. Such inconsistencies are now implicitly handled by _from_networkx;
                # #  if the annotation has any repeated (consecutive) nodes they are ignored, and if two consecutive nodes
                # #  don't have and edge in WS, they are connected by the shortest path. We should check that the returned
                # #  leg_route has distance and duration consistent with the annotation, otherwise raise a flag.
                # # leg_route = self._from_networkx(node_path)
                # leg_route = self._from_pgrouting(node_path)
                # # leg_route['osrm_edge_length'] = leg.annotation.distance
                # # leg_route['osrm_edge_duration'] = leg.annotation.duration
                # leg_route['leg'] = lix
                # leg_route.set_index(keys='leg', append=True, inplace=True)
                # route = route.append(leg_route)

                osrm_route_df = pd.DataFrame({'source': leg.annotation.nodes[:-1],
                                              'target': leg.annotation.nodes[1:],
                                              'length': leg.annotation.distance,
                                              'duration': leg.annotation.duration})
                osrm_route_df = osrm_route_df.loc[osrm_route_df.source != osrm_route_df.target]
                if osrm_route_df.empty:
                    continue  # fixme raise a flag when this happens
                unique_nodes = osrm_route_df.source.to_list() + [osrm_route_df.target.to_list()[-1]]

                def bounding_box(points):
                    y_coordinates, x_coordinates = zip(*points)
                    return {'xmin': min(x_coordinates), 'ymin': min(y_coordinates),
                            'xmax': max(x_coordinates), 'ymax': max(y_coordinates)}

                geom = polyline.decode(osrm_route.geometry)
                bbox = bounding_box(geom)
                # todo osm_id are unique_nodes[0] and  unique_nodes[-1] for src and tgt, but passing them will skip the
                #  id lookup
                src = PgrNode(id=None, osm_id=None, lon=geom[0][1], lat=geom[0][0])
                tgt = PgrNode(id=None, osm_id=None, lon=geom[-1][1], lat=geom[-1][0])
                via = [PgrNode(id=None, osm_id=n, lon=None, lat=None) for n in unique_nodes[1:-1]]

                bbox_nodes = [PgrNode(id=None, osm_id=None, lon=bbox['xmin'], lat=bbox['ymin']),
                              PgrNode(id=None, osm_id=None, lon=bbox['xmin'], lat=bbox['ymax']),
                              PgrNode(id=None, osm_id=None, lon=bbox['xmax'], lat=bbox['ymax']),
                              PgrNode(id=None, osm_id=None, lon=bbox['xmax'], lat=bbox['ymin'])]
                leg_route = self._network.shortest_path(src, tgt, via=via, bbox=bbox_nodes)
                # leg_route = leg_route.merge(right=osrm_route.rename(columns={'length': 'osrm_edge_length',
                #                                                              'duration': 'osrm_edge_duration'}),
                #                             how='left',
                #                             left_on=['source', 'target'],
                #                             right_on=['source', 'target'])
                leg_route['leg'] = lix
                leg_route['route'] = 0
                route = route.append(leg_route)

            return route

        else:

            for lix, leg in enumerate(osrm_route.legs):

                leg_route = pd.DataFrame({'source': leg.annotation.nodes[:-1],
                                          'target': leg.annotation.nodes[1:],
                                          'length': leg.annotation.distance,
                                          'duration': leg.annotation.duration,
                                          'speed': leg.annotation.speed})
                leg_route = leg_route.loc[leg_route.source != leg_route.target]
                if leg_route.empty:
                    continue  # fixme raise a flag when this happens
                geom = polyline.decode(osrm_route.geometry)
                geom_lat = [g[0] for g in geom]
                geom_lon = [g[1] for g in geom]
                leg_route['source_latitude'] = geom_lat[:-1]
                leg_route['source_longitude'] = geom_lon[:-1]
                leg_route['target_latitude'] = geom_lat[1:]
                leg_route['target_longitude'] = geom_lon[1:]
                leg_route['leg'] = lix
                leg_route['route'] = 0
                route = route.append(leg_route)

            return route

    def _from_here(self, json_response):
        waypoints = []
        position = 0.0
        for leg in json_response['response']['route'][0]['leg']:
            for index_link, link in enumerate(leg['link']):
                for index_shape, shape in enumerate(link['shape']):
                    if index_shape | ((not index_link) & (not index_shape)):
                        string = shape.split(',')
                        position = position + link['length']
                        waypoints.append({'latitude': float(string[0]),
                                          'longitude': float(string[1]),
                                          'distance': link['length'],
                                          'position': position,
                                          'speed_limit': link['speedLimit'] if 'speedLimit' in link else float('nan'),
                                          'traffic_speed': link['dynamicSpeedInfo']['trafficSpeed'],
                                          'base_speed': link['dynamicSpeedInfo']['baseSpeed'],
                                          'jam_factor': link['dynamicSpeedInfo']['jamFactor']
                                          })
        waypoints = pd.DataFrame(waypoints)
        links = pd.DataFrame(json_response['response']['route'][0]['leg'][0]['link'])
        links = pd.concat([links.drop(['dynamicSpeedInfo'], axis=1), links['dynamicSpeedInfo'].apply(pd.Series)],
                          axis=1)

        def shape_to_wkt(shape):
            shp_wkt = 'LINESTRING('
            for point in shape:
                lat, lon = point.split(',')
                shp_wkt += lon + ' ' + lat + ','
            shp_wkt = shp_wkt[:-1] + ')'
            return shp_wkt

        links['shape'] = links['shape'].apply(shape_to_wkt).map(shapely.wkt.loads)

        performance = json_response['response']['route'][0]['summary']
        key_map = {'baseTime': 'base_time', 'trafficTime': 'traffic_time', 'travelTime': 'travel_time'}
        for old_key, new_key in key_map.items():
            performance[new_key] = performance.pop(old_key)

        # here_dict = {'json': json_response, 'waypoints': waypoints, 'summary': performance}

        if self._matched:
            osrm_match = osrm.Match(trace=waypoints)  #, use_waypoints=True)
            if (isinstance(osrm_match.matches, osrm.MatchResult)) and (osrm_match.matches.code == 'Ok'):
                if len(osrm_match.matches.matchings) == 1:
                    matched_route = self._from_osrm(osrm_match.matches.matchings[0])
                    return matched_route
                else:
                    return [self._from_osrm(osrm_route) for osrm_route in osrm_match.matches.matchings]
            else:
                raise BaseException()  # fixme just output osrm error
        else:
            return links

    def plot(self):
        """

        """
        fig = go.Figure()
        fig.add_scattermapbox(
            mode="markers+lines",
            marker=go.scattermapbox.Marker(size=10),
            lon=self._nodes.longitude,
            lat=self._nodes.latitude,
            name='Route'
        )
        if self._here_dict is not None:
            fig.add_scattermapbox(
                mode="markers+lines",
                marker=go.scattermapbox.Marker(size=10),
                lon=self._here_dict['waypoints'].longitude,
                lat=self._here_dict['waypoints'].latitude,
                name='Route from Here'
            )
        fig.update_layout(
            showlegend=True,
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(
                    lat=self._nodes.latitude.dropna().mean(),
                    lon=self._nodes.longitude.dropna().mean()
                )
            )
        )
        fig.show()


class Engine:
    """Searches one or more routes based on the specified options

    :param bbid:
    :type bbid:
    :param mode:
    :type mode: str
    :param backend:
    :type backend: str
    :param network:
    :type network:
    :param traffic:
    :type traffic: bool
    :param terrain:
    :type terrain: bool
    :param weather:
    :type weather: bool
    """

    def __init__(self, bbid=None, mode: str = 'fastest', backend: str = 'here', network=None, traffic: bool = True,
                 terrain: bool = True, weather: bool = True):

        self._config = Config()

        # todo enable bbid from postgres
        if network is None and isinstance(bbid, str):
            self._network = Network(bbid=bbid)
        elif isinstance(network, Network):
            self._network = network
        else:
            raise Exception("either network or bbid must be valid")

        if mode in ['fastest', 'shortest']:
            self._mode = mode
        elif mode is None:
            self._mode = 'fastest'
        else:
            raise ValueError("Invalid value " + mode + " for argument mode")

        if backend in ['here', 'ws', 'osrm', 'mapbox']:
            self._backend = backend
        elif backend is None:
            self._backend = 'here'
        else:
            raise ValueError("Invalid value " + backend + " for argument backend")

        assert isinstance(traffic, bool)
        self._traffic = traffic
        assert isinstance(terrain, bool)
        self._terrain = terrain
        assert isinstance(weather, bool)
        self._weather = weather

    def route(self, origin: Origin, destination: Destination, mode: str = None, backend: str = None,
              matched: bool = True) -> RouteClient:
        """

        :param origin:
        :type origin: Origin
        :param destination:
        :type destination: Destination
        :param mode:
        :type mode: str
        :param backend:
        :type backend: str
        :param matched:
        :type matched: bool
        :return:
        :rtype: RouteClient
        """

        if mode in ['fastest', 'shortest']:
            self._mode = mode
        elif mode is not None:
            warnings.warn("Invalid value " + str(mode) + " for argument mode, defaulting to " + self._mode)

        if backend in ['here', 'ws', 'osrm', 'mapbox']:
            self._backend = backend
        elif backend is not None:
            warnings.warn("Invalid value " + str(backend) + " for argument backend, defaulting to " + self._backend)

        if self._backend == 'here':
            return self._route_here(origin, destination, matched)
        if self._backend is 'mapbox':
            return self._route_mapbox(origin, destination)
        if self._backend is 'osrm':
            return self._route_osrm(origin, destination, matched)
        if self._backend is 'ws':
            return self._route_ws(origin, destination)

    def _route_ws(self, origin, destination):
        """

        :param origin:
        :type origin: Origin
        :param destination:
        :type destination: Destination
        :return:
        :rtype: RouteClient
        """
        raise NotImplementedError
        # weight = "length"
        # path = nx.shortest_path(self._network.graph, origin.id, destination.id, weight=weight)
        # return RouteClient(from_networkx=path)

    def _route_here(self, origin: Origin, destination: Destination, matched: bool) -> RouteClient:
        """

        :param origin:
        :type origin: Origin
        :param destination:
        :type destination: Destination
        :return:
        :rtype: RouteClient
        """

        # construct request
        if self._traffic:
            mode_string = self._mode + ";car;traffic:enabled"
        else:
            mode_string = self._mode + ";car;traffic:disabled"
        waypoint0 = "geo!{},{}".format(origin.latitude, origin.longitude)
        waypoint1 = "geo!{},{}".format(destination.latitude, destination.longitude)
        params = {
            'app_id': self._config.here['app_id'],
            'app_code': self._config.here['app_code'],
            'mode': mode_string,
            'waypoint0': waypoint0,
            'waypoint1': waypoint1,
            'representation': 'navigation',
            'linkAttributes': 'sh,ds,sl,le',  # shape, dynamic speed info, speed limits, length
            'maneuverAttributes': 'di,le',  # direction, length
            'routeAttributes': 'sm'  # summary
        }
        url = "https://route.api.here.com/routing/7.2/calculateroute.json"

        # make request
        response = requests.get(url, params)
        if response.ok:
            json_response = response.json()
        else:
            raise BaseException("Error getting route {} {}".format(str(response.status_code), response.content))
        return RouteClient(from_here=json_response, network=self._network, traffic=False, weather=self._weather,
                           terrain=self._terrain, matched=matched)

    def _route_osrm(self, origin, destination, matched):
        """

        :param origin:
        :type origin: Origin
        :param destination:
        :type destination: Destination
        :return:
        :rtype: RouteClient
        """
        router = osrm.RouteService()
        response = router._request(origin.longitude, origin.latitude, destination.longitude, destination.latitude)
        # fixme there may be multiple routes
        return RouteClient(from_osrm=response.routes[0], network=self._network, traffic=False, weather=self._weather,
                           terrain=self._terrain, matched=matched)

    def _route_mapbox(self, origin, destination):
        """

        :param origin:
        :type origin: Origin
        :param destination:
        :type destination: Destination
        :return:
        :rtype: RouteClient
        """
        raise NotImplementedError


class Route(models.Base, models.ConnectedModel):
    """
    A route is a proposed or traveled route through a map.
    """
    __tablename__ = 'route'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    traffic_id = Column(Integer, ForeignKey('traffic.id'))
    traffic = relationship(Traffic)


class WayPoint(models.Base, models.ConnectedModel):
    """ """
    __tablename__ = 'way_point'
    id = Column(Integer, primary_key=True)
    route_id = Column(Integer, ForeignKey('route.id'))
    route = relationship(Route)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    order = Column(Integer, nullable=False)


class RouteSegment(models.Base, models.ConnectedModel):
    """ """
    __tablename__ = 'route_segment'
    id = Column(Integer, primary_key=True)
    way_point_start_id = Column(Integer, ForeignKey('way_point.id'))
    way_point_start = relationship(WayPoint, foreign_keys=[way_point_start_id])
    wap_point_end_id = Column(Integer, ForeignKey('way_point.id'))
    way_point_end = relationship(WayPoint, foreign_keys=[wap_point_end_id])

