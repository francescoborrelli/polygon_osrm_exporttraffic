#  Copyright (c) 2020. AV Connect Inc.
"""This module provides map matching functionality. """

import ws_maps.network as nw
import ws_maps.osrm as osrm
import ws_maps.config as cfg
import pandas as pd
import numpy as np
import scipy as sp
import scipy.integrate
import plotly.graph_objects as go
import plotly.io as pio
import warnings
import polyline
from ws_maps.pgrouting import PgrNode

pio.renderers.default = "browser"
mapbox_access_token = cfg.Config.mapbox


class Match(object):
    """Map matching class used to match GPS traces to a map.

    :param trace:
    :type trace:
    :param network:
    :type network:
    :param bbid:
    :type bbid:
    :param backend:
    :type backend:
    :param columns:
    :type columns:
    """
    def __init__(self, trace=None, network=None, bbid='chino', backend='osrm',
                 columns={'latitude': 'latitude', 'longitude': 'longitude', 'time': 'time', 'speed': 'speed'}):
        if trace is None:
            raise Exception("trace cannot be empty")
        if not isinstance(trace, pd.DataFrame):
            raise Exception("trace must be a dataframe")
        self._trace = trace  # [44:205]
        if not isinstance(columns, dict):
            raise Exception
        else:
            self._columns = columns
        self._filtered_trace = self._filter_trace(trace)
        if network is None and isinstance(bbid, str):
            self._network = nw.Network(bbid=bbid)
        elif isinstance(network, nw.Network):
            self._network = network
        else:
            raise Exception("either network or bbid must be valid")
        if backend == 'osrm':
            # todo check osrm is running with an appropriate network file
            self._backend = backend
        else:
            # todo connect all-python matching https://github.com/simonscheider/mapmatching
            warnings.warn(backend + " not supported, defaulting to osrm")
            self._backend = 'osrm'

        self._osrm_match = osrm.Match(trace=self._filtered_trace)
        self._osrm_match.plot()
        # fixme better handle case of no match found
        self._route, self._trajectory = self._osrm_to_ws(self._osrm_match.matches, self._osrm_match.trace_index)
        self._arcs = self._match_to_arcs(self._route, self._trajectory, pandas=True)

    def _filter_trace(self, df):
        fdf = df.rename(columns=self._columns)
        fdf = fdf.set_index(fdf['time']).dropna()
        return fdf

    @property
    def route(self):
        """The matched route, defined as the paths of visited nodes and edges.

        :return:
        """
        return self._route

    @property
    def trajectory(self):
        """The matched trajectory, containing the GPS coordinates and their corresponding values snapped to the map.

        :return:
        :rtype:
        """
        return self._trajectory

    @property
    def arcs(self):
        """The matched arcs.

        :return:
        :rtype:
        """
        return self._arcs

    def _osrm_to_ws(self, matches, trace_index):
        """Converts the matching results from the OSRM format to the WideSense format

        :return:
        """
        assert(isinstance(matches, osrm.MatchResult))
        assert(matches.code == 'Ok')

        route = self._matchings_to_route(self._network, matches.matchings)
        trajectory = self._tracepoints_to_trajectory(self._filtered_trace, matches.tracepoints, route, trace_index)
        # todo add osrm summaries validation
        return route, trajectory

    @staticmethod
    def _matchings_to_route(network, matchings, way_ids=None, lengths=None):
        """Converts a list of OSRM matchings (Route objects) to a corresponding WideSense route

        :return:
        """

        # todo make this a call to Route(from_osrm=...), so it returns an actual route and can optionally pull traffic
        #  terrain and weather; notice that in general we'd return more than one route, unlike now

        routes = pd.DataFrame()
        for rix, matching_route in enumerate(matchings):
            if matching_route.confidence < 0.1:
                continue
            for lix, leg in enumerate(matching_route.legs):
                osrm_route = pd.DataFrame({'source': leg.annotation.nodes[:-1],
                                           'target': leg.annotation.nodes[1:],
                                           'length': leg.annotation.distance,
                                           'duration': leg.annotation.duration})
                osrm_route = osrm_route.loc[osrm_route.source != osrm_route.target]
                if osrm_route.empty:
                    continue  # fixme raise a flag when this happens
                unique_nodes = osrm_route.source.to_list() + [osrm_route.target.to_list()[-1]]

                def bounding_box(points):
                    y_coordinates, x_coordinates = zip(*points)
                    return {'xmin': min(x_coordinates), 'ymin': min(y_coordinates),
                            'xmax': max(x_coordinates), 'ymax': max(y_coordinates)}

                geom = polyline.decode(matching_route.geometry)
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
                leg_route = network.shortest_path(src, tgt, via=via, bbox=bbox_nodes)
                # leg_route = leg_route.merge(right=osrm_route.rename(columns={'length': 'osrm_edge_length',
                #                                                              'duration': 'osrm_edge_duration'}),
                #                             how='left',
                #                             left_on=['source', 'target'],
                #                             right_on=['source', 'target'])
                leg_route['leg'] = lix
                leg_route['route'] = rix
                routes = routes.append(leg_route)

        routes = routes.reset_index(drop=True)

        if not routes.empty:
            way_path = routes['way'].dropna().tolist()
            node_path = []
            for ix, group in routes.groupby(['route']):
                group_node_path = group.source.tolist() + [group.target.tolist()[-1]]
                node_path += group_node_path
        else:
            way_path = []
            node_path = []

        return {'ways': network.ways(ids=way_path),
                'nodes': network.nodes(ids=node_path),
                'df': routes}

    @staticmethod
    def _tracepoints_to_trajectory(trace, tracepoints, route, trace_index):
        """Converts a list of OSRM tracepoints (Waypoint objects) to a corresponding WideSense trajectory

        :return:
        """

        trajectory = pd.DataFrame(
            columns=['gps_latitude', 'gps_longitude', 'gps_time', 'datetime', 'matched_latitude', 'matched_longitude',
                     'matching_index', 'waypoint_index', 'edge_index', 'matched_edge', 'matched_source',
                     'matched_target'])
        trajectory.gps_latitude = trace.latitude
        trajectory.gps_longitude = trace.longitude
        trajectory.gps_time = trace.index
        trajectory.datetime = trace.datetime
        trajectory.matched_longitude = [tr.location[0] if tr is not None else None for tr in tracepoints]
        trajectory.matched_latitude = [tr.location[1] if tr is not None else None for tr in tracepoints]
        trajectory.matching_index = trace_index.route.values
        trajectory.waypoint_index = trace_index.leg.values
        trajectory.edge_index = trace_index.edge.values
        trajectory.matched_source = trace_index.source.values
        trajectory.matched_target = trace_index.target.values
        # idx = pd.IndexSlice
        # lookup = route['df'].set_index(['source', 'target'])
        # def find_way(row, lookup, idx):
        #     try:
        #         way = lookup.loc[idx[row.matched_source, row.matched_target], 'way']
        #         if isinstance(way, pd.Series):
        #             way = way.values[0]
        #     except:
        #         warnings.warn("matching/waypoint index may be inconsistent here..." + str(row.index))
        #         way = None
        #     return way
        # trajectory.matched_way = trajectory.apply(lambda x: find_way(x, lookup, idx), axis=1)
        return trajectory

    def _match_to_arcs(self, route, trajectory, pandas=True):
        """

        :return:
        """

        arcs = pd.DataFrame(columns=['route_order', 'leg_order', 'edge_order', 'osm_id', 'osm_source_id',
                                     'osm_target_id', 'bbid_id', 'map_length'])
        if not route['df'].empty:
            arcs['route_order'] = route['df'].route.values  # rix,
            arcs['leg_order'] = route['df'].leg.values  # lix,
            arcs['edge_order'] = route['df'].index  # eix,
            arcs['osm_id'] = route['df'].way.values
            arcs['osm_source_id'] = route['df'].source.values
            arcs['osm_target_id'] = route['df'].target.values
            arcs['start_lat'] = self._network.nodes(ids=arcs.osm_source_id.to_list(), columns=['latitude']).values
            arcs['start_lon'] = self._network.nodes(ids=arcs.osm_source_id.to_list(), columns=['longitude']).values
            arcs['end_lat'] = self._network.nodes(ids=arcs.osm_source_id.to_list(), columns=['latitude']).values
            arcs['end_lon'] = self._network.nodes(ids=arcs.osm_source_id.to_list(), columns=['longitude']).values
            arcs['bbid_id'] = ''
            arcs['map_length'] = route['df'].length  #osrm_edge_length

            for ix, group in arcs.groupby(['route_order']):
                group_datetimes = trajectory.loc[trajectory.matching_index == group.route_order.values[0],
                                                 'datetime'].values
                trace = self._trace.loc[(group_datetimes[0] <= self._trace.datetime) &
                                        (group_datetimes[-1] >= self._trace.datetime)]
                trace = trace.set_index(trace.index.values - trace.index.values[0])
                trace.time = trace.time.values - trace.time.values[0]

                # map_distance = np.insert(route['df'].loc[
                #                              route['df'].route == group.route_order.values[
                #                                  0], 'osrm_edge_length'].cumsum().values, 0, 0.0)
                map_distance = np.insert(route['df'].loc[
                    route['df'].route == group.route_order.values[0], 'length'].cumsum().values, 0, 0.0)
                meas_distance = sp.integrate.cumtrapz(x=trace.time, y=trace.speed, initial=0.0)
                factor = (map_distance[-1] - map_distance[0]) / (meas_distance[-1] - meas_distance[0])
                meas_distance = meas_distance * factor
                node_times = np.interp(map_distance, meas_distance, trace.index.total_seconds())
                arcs.loc[group.index, 'start_time'] = group_datetimes[0] + pd.to_timedelta(node_times[:-1], unit='S')
                arcs.loc[group.index, 'end_time'] = group_datetimes[0] + pd.to_timedelta(node_times[1:] - 1.0e-9, unit='S')
                arcs.loc[group.index, 'distance_factor'] = factor

        # add unmatched of low confidence segments
        for ix, group in trajectory.groupby(['matching_index']):
            if group.matching_index.iloc[0] not in arcs.route_order.unique():
                arcs = arcs.append({'route_order': group.matching_index.iloc[0],
                                    'start_time': group.datetime.iloc[0],
                                    'end_time': group.datetime.iloc[-1],
                                    'leg_order': None,
                                    'edge_order': None,
                                    'osm_id': None,
                                    'osm_source_id': None,
                                    'osm_target_id': None,
                                    'bbid_id': '',
                                    'map_length': None,
                                    'start_lat': group.gps_latitude.iloc[0],
                                    'start_lon': group.gps_longitude.iloc[0],
                                    'end_lat': group.gps_latitude.iloc[-1],
                                    'end_lon': group.gps_longitude.iloc[-1],
                                    'distance_factor': 1.0},
                                   ignore_index=True)

        arcs.start_time = arcs.start_time.dt.tz_localize('UTC')
        arcs.end_time = arcs.end_time.dt.tz_localize('UTC')
        arcs = arcs.sort_values(by=['start_time'])
        arcs = arcs.reset_index(drop=True)

        # if pandas:
        #     return pd.DataFrame(arcs)
        # else:
        return arcs

    def plot(self, display=True):
        """Plot the measured and matched coordinates on a map.

        :param display:
        :return:
        """
        data = list([
            go.Scattermapbox(
                mode="lines+markers",
                marker=go.scattermapbox.Marker(size=14),
                lon=self._trace.longitude,
                lat=self._trace.latitude,
                name='measured gps trace'
            ),
            # go.Scattermapbox(
            #     mode="lines+markers",
            #     marker=go.scattermapbox.Marker(size=14),
            #     lon=self._trajectory.gps_longitude,
            #     lat=self._trajectory.gps_latitude,
            #     name='gps output from osrm'
            # ),
            go.Scattermapbox(
                mode="lines+markers",
                marker=go.scattermapbox.Marker(size=14),
                lon=self._trajectory.matched_longitude,
                lat=self._trajectory.matched_latitude,
                name='gps trace snapped to road network'
            ),
            go.Scattermapbox(
                mode="markers",
                marker=go.scattermapbox.Marker(size=14),
                lon=self._route['nodes'].longitude,
                lat=self._route['nodes'].latitude,
                name='matched road network nodes'
            )
        ])
        layout = go.Layout(
            showlegend=True,
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(
                    lat=self._trace.latitude.dropna().mean(),
                    lon=self._trace.longitude.dropna().mean()
                )
            )
        )
        if display:
            fig = go.Figure(data=data, layout=layout)
            fig.show()
        else:
            return data, layout
