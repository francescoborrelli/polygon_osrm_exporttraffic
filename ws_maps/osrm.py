#  Copyright (c) 2020. AV Connect Inc.
"""This module provides an interface to osrm-backend. """

from ws_maps.config import Config
import pandas as pd
import json
import functools
import requests
import warnings
from itertools import groupby
import plotly.graph_objects as go
import plotly.io as pio
import polyline
import numpy as np
import copy

pio.renderers.default = "browser"
mapbox_access_token = Config().mapbox


class Result(object):
    """Wrapper for result objects off the OSRM HTTP API

    Deserialize JSON responses into objects rather than dictionaries, simplifying results validation and manipulation.
    """

    _fields = []

    @staticmethod
    def _init_arg(expected_type, islist, value):
        if islist and isinstance(value, list):
            vlist = []
            for v in value:
                if isinstance(v, expected_type) or v is None:
                    vlist.append(v)
                elif type(v) in [float, int] and expected_type in [float, int]:
                    vlist.append(expected_type(v))
                else:
                    vlist.append(expected_type(**v))
            return vlist
        else:
            if isinstance(value, expected_type):
                return value
            elif type(value) in [float, int] and expected_type in [float, int]:
                return expected_type(value)
            elif type(value) is str:
                pass  # todo remove
            elif value is None:
                return None
            else:
                return expected_type(**value)

    def __init__(self, **kwargs):
        field_names, list_flags, field_types, optional_flags = zip(*self._fields)
        assert([isinstance(name, str) for name in field_names])
        assert ([isinstance(flag, bool) for flag in list_flags])
        assert([isinstance(type_, type) for type_ in field_types])
        assert ([isinstance(flag, bool) for flag in optional_flags])

        for name, is_list, field_type, is_optional in self._fields:
            if name in kwargs.keys():
                setattr(self, name, self._init_arg(field_type, is_list, kwargs.pop(name)))
            elif is_optional:
                continue
            else:
                raise KeyError('Key {} was not found'.format(name))

        # Check for any remaining unknown arguments
        # if kwargs:
        #     warnings.warn('Invalid arguments(s): {}'.format(','.join(kwargs)))


class Annotation(Result):
    """Annotation of the whole route leg with fine-grained information about each segment or node id.

    Attributes:
        distance: The distance, in metres, between each pair of coordinates
        duration: The duration between each pair of coordinates, in seconds
        datasources: The index of the datasource for the speed between each pair of coordinates. 0 is the default
            profile, other values are supplied via --segment-speed-file to osrm-contract
        nodes: The OSM node ID for each coordinate along the route, excluding the first/last user-supplied coordinates
        speed (list of float)
        metadata (dict)
        weight (list of float)
    """

    _fields = [('distance', True, float, False),
               ('duration', True, float, False),
               ('datasources', True, int, False),
               ('nodes', True, int, False),
               ('speed', True, float, False)]


class StepManeuver(Result):
    """Step maneuver

    Attributes
        location: A [longitude, latitude] pair describing the location of the turn.
        bearing_before: The clockwise angle from true north to the direction of travel immediately before the maneuver.
        bearing_after: The clockwise angle from true north to the direction of travel immediately after the maneuver.
        type: A string indicating the type of maneuver. New identifiers might be introduced without API change. Types
            unknown to the client should be handled like the turn type, the existence of correct modifier values is
            guaranteed.
    """

    _fields = [('location', True, float, False),  # location is a coordinate (lon/lat)
               ('bearing_before', False, float, False),
               ('bearing_after', False, float, False),
               ('type', False, str, False),  # type takes a limited number of values
               ('modifier', False, str, True),  # modifier takes a limited number of values
               ('exit', False, int, True)]


class Lane(Result):
    """A Lane represents a turn lane at the corresponding turn location.

        Attributes
            indications: a indication (e.g. marking on the road) specifying the turn lane. A road can have multiple
                indications (e.g. an arrow pointing straight and left). The indications are given in an array, each
                containing one of the following types. Further indications might be added on without an API version
                change.
            valid: a boolean flag indicating whether the lane is a valid choice in the current maneuver

    """

    _fields = [('indications', True, str, False),  # indications takes a limited number of values
               ('valid', False, bool, False)]


class Intersection(Result):
    """An intersection gives a full representation of any cross-way the path passes bay.

    For every step, the very first intersection (intersections[0]) corresponds to the location of the StepManeuver.
    Further intersections are listed for every cross-way until the next turn instruction.

    Attributes
        location: A [longitude, latitude] pair describing the location of the turn.
        bearings: A list of bearing values (e.g. [0,90,180,270]) that are available at the intersection. The bearings
            describe all available roads at the intersection.
        entry: A list of entry flags, corresponding in a 1:1 relationship to the bearings. A value of true indicates
            that the respective road could be entered on a valid route. false indicates that the turn onto the
            respective road would violate a restriction.
        in: index into bearings/entry array. Used to calculate the bearing just before the turn. Namely, the clockwise
            angle from true north to the direction of travel immediately before the maneuver/passing the intersection.
            Bearings are given relative to the intersection. To get the bearing in the direction of driving, the bearing
            has to be rotated by a value of 180. The value is not supplied for depart maneuvers.
        out: index into the bearings/entry array. Used to extract the bearing just after the turn. Namely, The clockwise
            angle from true north to the direction of travel immediately after the maneuver/passing the intersection.
            The value is not supplied for arrive maneuvers.
        lanes: Array of Lane objects that denote the available turn lanes at the intersection. If no lane information is
            available for an intersection, the lanes property will not be present.
        classes (list of str)
    """

    _fields = [('location', True, float, False),  # location is a coordinate (lon/lat)
               ('bearings', True, float, False),
               ('entry', True, int, False),
               ('in', False, int, True),
               ('out', False, int, True),
               ('lanes', True, Lane, True)]


class RouteStep(Result):
    """A step consists of a maneuver such as a turn or merge.

     A step consists of a maneuver such as a turn or merge, followed by a distance of travel along a single way to the
     subsequent step.

    Attributes:
        distance: The distance of travel from the maneuver to the subsequent step, in float meters.
        duration: The estimated travel time, in float number of seconds.
        geometry: The unsimplified geometry of the route segment, depending on the geometries parameter.
        name: The name of the way along which travel proceeds.
        ref: A reference number or code for the way. Optionally included, if ref data is available for the given way.
        pronunciation: The pronunciation hint of the way name. Will be undefined if there is no pronunciation hit.
        destinations: The destinations of the way. Will be undefined if there are no destinations.
        mode: A string signifying the mode of transportation.
        maneuver: A StepManeuver object representing the maneuver.
        intersections: A list of Intersection objects that are passed along the segment, the very first belonging to the
            StepManeuver
        weight (int)
        driving_side (str)
    """

    _fields = [('distance', False, float, False),
               ('duration', False, float, False),
               ('geometry', False, str, False),
               ('name', False, str, False),
               ('ref', False, str, True),
               ('pronunciation', False, str, True),  # will be undefined if there is no pronunciation hit.
               ('destinations', False, str, True),  # will be undefined if there are no destinations.
               ('mode', False, str, False),
               ('maneuver', False, StepManeuver, False),
               ('intersections', True, Intersection, False)]


class RouteLeg(Result):
    """Represents a route between two waypoints.

    Attributes:
        distance: The distance traveled by this route leg, in float meters.
        duration: The estimated travel time, in float number of seconds.
        summary: Summary of the route taken as string. Depends on the steps parameter:
        steps: Depends on the steps parameter.
        annotation: Additional details about each coordinate along the route geometry:
        weight (int):
    """

    _fields = [('distance', False, float, False),
               ('duration', False, float, False),
               ('summary', False, str, False),
               ('steps', True, RouteStep, True),  # empty array if the step parameter is false
               ('annotation', False, Annotation, True)]  # undefined if the annotations parameter is false


class Route(Result):
    """Represents a route through (potentially multiple) waypoints.

    Attributes:
        distance: The distance traveled by the route, in float meters.
        duration: The estimated travel time, in float number of seconds.
        geometry: The whole geometry of the route value depending on overview parameter, format depending on the
            geometries parameter. See RouteStep's geometry field for a parameter documentation.
        legs: The legs between the given waypoints, an array of RouteLeg objects.
        confidence (float):
        weight (int):
        weight_name (str):
    """

    _fields = [('distance', False, float, False),
               ('duration', False, float, False),
               ('geometry', False, str, False),
               ('legs', True, RouteLeg, False),
               ('confidence', False, float, True)]  # confidence only present when doing matching


class Waypoint(Result):
    """Object used to describe waypoint on a route.

    Attributes:
        name: Name of the street the coordinate snapped to
        location: Array that contains the [longitude, latitude] pair of the snapped coordinate
        distance: The distance of the snapped point from the original
        hint: Unique internal identifier of the segment (ephemeral, not constant over data updates) This can be used on
            subsequent request to significantly speed up the query and to connect multiple services. E.g. you can use
            the hint value obtained by the nearest query as hint values for route inputs.
    """

    _fields = [('name', False, str, False),
               ('location', True, float, False),  # location is a coordinate (lon/lat)
               ('distance', False, float, False),
               ('hint', False, str, False),
               ('matchings_index', False, int, True),  # only present when doing matching
               ('waypoint_index', False, int, True),  # only present when doing matching
               ('alternatives_count', False, int, True)]  # only present when doing matching


class MatchResult(Result):
    """

    Attributes:
        code: if the request was successful Ok otherwise see the service dependent and general status codes.
        tracepoints: Array of Waypoint objects representing all points of the trace in order. If the trace point was
            omitted by map matching because it is an outlier, the entry will be null. Each Waypoint object has the
            following additional properties:
        matchings: An array of Route objects that assemble the trace. Each Route object has the following additional
            properties:
        confidence: Confidence of the matching. float value between 0 and 1. 1 is very confident that the matching is
            correct.
    """

    _fields = [('code', False, str, False),
               ('tracepoints', True, Waypoint, False),
               ('matchings', True, Route, False)]


class RouteResult(Result):
    """

    Attributes:
        code: if the request was successful Ok otherwise see the service dependent and general status codes.
        waypoints: Array of Waypoint objects representing all waypoints in the request.
        routes: An array of Route objects.
    """

    _fields = [('code', False, str, False),
               ('waypoints', True, Waypoint, False),
               ('routes', True, Route, False)]


class Match:
    """Wrapper class for the Match service of the OSRM HTTP API

    """
    def __init__(self, trace):
        self._config = Config()
        if isinstance(trace, pd.DataFrame):
            self._trace = trace.to_dict('records')
            self._matches = None
            self._trace = self._trace
            matches_dict = self._request(use_waypoints=False)
            if matches_dict['code'] == 'Ok':
                # todo assert response complies with assumptions
                matches = MatchResult(**matches_dict)
                self._matches = self._match_sub_routes(matches)
                self.trace_index = self._stitch_route_legs(matches)
            else:
                self._matches = MatchResult(**{'code': 'Ok', 'tracepoints': [None] * len(self._trace), 'matchings': [],
                                               'confidence': 1.0})
                self.trace_index = pd.DataFrame(
                    columns=['route', 'leg', 'edge', 'source', 'target', 'distance_steps', 'distance_annotation',
                             'duration_steps', 'duration_annotation'],
                    index=range(0, len(self._trace))
                )
                self.trace_index.route = -1
        else:
            # todo raise type error if not dataframe
            self._trace = None
            self._matches = None
            self.trace_index = None

    def _request(self, trace=None, use_waypoints=False, waypoints=None):
        """

        """
        if trace is None:
            trace = self._trace
        points = functools.reduce(lambda m, x: m + str(x['longitude']) + "," + str(x['latitude']) + ";", trace, "")[:-1]
        # timestamps = functools.reduce(lambda m, x: m + "{:.0f};".format(x['timestamp']), trace, "")[:-1]
        radiuses = ";".join(map(str, [20.0]*len(trace)))  # todo adapt based on the actual accuracy
        params = {
            'steps': 'true',
            'geometries': 'polyline',  # use geojson?
            'overview': 'full',
            'annotations': 'true',
            'radiuses': radiuses,
            'tidy': 'false',
            # 'timestamps': timestamps
        }
        # The waypoints parameter reduces the number of route legs returned. By default every gps sample is considered
        # a waypoint, the waypoints parameter is a list of which points in the trace are to be considered as waypoints.
        # See the full discussion here https://github.com/Project-OSRM/osrm-backend/issues/4669
        # In our typical use case we always want to only use the first and last points in the trace as waypoints;
        # however with this setting OSRM will error if some trace points are not matched, or if the trace has to be
        # split, which could happen if we have short drops in data. Workarounds include prefiltering the trace and we
        # calling the matching twice, first on the entire trace without setting the waypoints parameter, then only on
        # the part(s) of the trace that could be matched in the first pass.
        if use_waypoints:
            if isinstance(waypoints, list):
                params['waypoints'] = functools.reduce(lambda m, x: m + str(x) + ";", waypoints, "") + str(len(trace)-1)
            else:
                params['waypoints'] = '0;' + str(len(trace)-1)
        url = "http://{}:{}/match/v1/car/".format(self._config.osrm['host'], self._config.osrm['port']) + points + "?"
        response = requests.get(url, params)
        return json.loads(response.content)

    def _match_sub_routes(self, matches):
        # in most cases, matching_index = [t if t is None else t.matchings_index for t in matches.tracepoints]
        # however, OSRM sometimes returns (short) sequences of None tracepoints followed and preceded by
        # tracepoints with the same matching_index (I've seen it happen with up to consecutive 2 samples).
        # It looks like these are short sequences of (noisy) GPS samples that are far from the current match, but are
        # not enough to break the matched route because the following samples go back on track; they are probably
        # assumed as plausible GPS noise by the hmm.
        # To distinguish this case from the common case of None tracepoints followed by a new route, we interpolate the
        # matching_index: if the matching_index changes before and after, the interpolated values will be not integer
        # and are set back to None, otherwise they are maintained
        matchings_index = pd.Series([t if t is None else t.matchings_index for t in matches.tracepoints])\
                          .interpolate(limit_area='inside', limit=999)
        matchings_index = matchings_index.where(matchings_index % 1 < 1.0e-3, None).to_list()

        data = zip(self._trace, matchings_index)
        grouped_data = [list(g) for k, g in groupby(data, lambda x: x[1])]
        submatches = []
        for subdata in grouped_data:
            subtrace, subtracepoints = list(zip(*subdata))
            if subtracepoints[0] is None:
                submatches.append({'tracepoints': [None] * len(subtrace)})
            else:
                subwaypoints = []
                for v in set(subtracepoints):
                    if subtracepoints.index(v) == 0:
                        subwaypoints.append(subtracepoints.index(v))
                    else:
                        subwaypoints += [subtracepoints.index(v) - 1, subtracepoints.index(v)]
                submatches_dict = self._request(trace=subtrace, use_waypoints=True, waypoints=subwaypoints)
                if submatches_dict['code'] == 'Ok':
                    submatches.append(MatchResult(**submatches_dict))
                else:
                    warnings.warn('something went wrong')
                    submatches.append({'tracepoints': [None] * len(subtrace)})
        return self._stitch_sub_matches(submatches)

    def _stitch_sub_matches(self, submatches):
        match = None
        for sm in submatches:
            if isinstance(sm, MatchResult):
                match = copy.deepcopy(sm)
                break
        if match is None:
            return None
        match.code = ''
        match.tracepoints = []
        match.matchings = []
        match_idx = 0
        for match_result in submatches:
            if isinstance(match_result, dict) and 'tracepoints' in match_result.keys():
                match.tracepoints += match_result['tracepoints']
            elif isinstance(match_result, MatchResult):
                match.code = 'Ok'
                tracepoints = match_result.tracepoints.copy()
                for waypoint in tracepoints:
                    try:
                        waypoint.matchings_index = match_idx
                    except AttributeError:
                        continue
                        # fixme this happens when there's a short unmatched segment (I've seen it happen with
                        #  up to consecutive 2 samples) that does not cause the route to break - probably
                        #  assumed as plausible GPS noise by the hmm - shows up as None tracepoints with same
                        #  matching before and after
                match.tracepoints += tracepoints
                match.matchings += match_result.matchings
                match_idx += 1
            else:
                raise TypeError("list elements must be either a MatchResult or None; trying with default")
        assert (match.code == 'Ok')
        return match

    def _stitch_route_legs(self, match):
        """
        patch to deal with this issue https://github.com/Project-OSRM/osrm-backend/issues/5490
        The logic is roughly:

            Append the first leg annotations to your result.
            Repeat the following for each extra leg:
            2a. Trim the first two nodes from the start of the next leg annotation to append
            2b. Append the trimmed leg annotation to your result
        """

        def _merge(a, b):
            mtchs = (i for i in range(len(b), 0, -1) if b[:i] == a[-i:])
            i = next(mtchs, 0)
            return a + b[i:]

        trace_index = pd.DataFrame(columns=['route', 'leg', 'edge', 'source', 'target', 'distance_steps',
                                            'distance_annotation', 'duration_steps', 'duration_annotation'])
        trace_index.route = [int(t.matchings_index) if t is not None else None for t in match.tracepoints]
        tix_list = iter(trace_index.route.dropna().index)
        tix = next(tix_list)
        for rix, route in enumerate(match.matchings):
            route_geometry = []
            route_nodes = []
            route_distance_steps = 0.0
            route_distance_annotation = []
            route_duration_steps = 0.0
            route_duration_annotation = []
            for lix, leg in enumerate(route.legs):
                if lix == 0:  # rix == 0 and lix == 0:
                    route_nodes = copy.copy(leg.annotation.nodes)
                    route_distance_annotation = copy.copy(leg.annotation.distance)
                    route_duration_annotation = copy.copy(leg.annotation.duration)
                elif (len(leg.annotation.nodes) == 2) and (leg.annotation.nodes[0] == route_nodes[-1]):
                    # not clear why, but it sometimes happens that when the tracepoint is in a new edge, there are only
                    # 2 nodes in the annotation instead of 3 as usual; I suppose it could be because the previous
                    # tracepoint is very close to its target node.
                    route_nodes += copy.copy(leg.annotation.nodes[1:])
                    route_distance_annotation += copy.copy(leg.annotation.distance)
                    route_duration_annotation += copy.copy(leg.annotation.duration)
                else:
                    route_nodes += copy.copy(leg.annotation.nodes[2:])  # _merge(leg_nodes, leg.annotation.nodes)
                    route_distance_annotation += copy.copy(leg.annotation.distance)
                    route_duration_annotation += copy.copy(leg.annotation.duration)
                for six, step in enumerate(leg.steps):
                    if lix == 0 and six == 0:  # rix == 0 and
                        route_geometry = polyline.decode(step.geometry)
                    else:
                        route_geometry = _merge(route_geometry, polyline.decode(step.geometry))
                    route_distance_steps += step.distance
                    route_duration_steps += step.duration

                trace_index.at[tix, 'route'] = int(rix)
                trace_index.at[tix, 'leg'] = 0
                # todo this can make the first edge non 0 if the first annotation has more than 2 nodes
                trace_index.at[tix, 'edge'] = len(route_nodes) - 2
                trace_index.at[tix, 'source'] = route_nodes[-2]
                trace_index.at[tix, 'target'] = route_nodes[-1]
                trace_index.at[tix, 'distance_steps'] = route_distance_steps
                trace_index.at[tix, 'duration_steps'] = route_duration_steps
                trace_index.at[tix, 'distance_annotation'] = sum(route_distance_annotation)
                trace_index.at[tix, 'duration_annotation'] = sum(route_duration_annotation)
                tix = next(tix_list)
            trace_index.at[tix, 'route'] = int(rix)
            trace_index.at[tix, 'leg'] = 0
            trace_index.at[tix, 'edge'] = len(route_nodes) - 2
            trace_index.at[tix, 'source'] = route_nodes[-2]
            trace_index.at[tix, 'target'] = route_nodes[-1]
            trace_index.at[tix, 'distance_steps'] = route_distance_steps
            trace_index.at[tix, 'duration_steps'] = route_duration_steps
            trace_index.at[tix, 'distance_annotation'] = sum(route_distance_annotation)
            trace_index.at[tix, 'duration_annotation'] = sum(route_duration_annotation)
            try:
                tix = next(tix_list)
            except StopIteration:
                pass
        # now there's a single merged leg for this route

        # numerate unmatched segments
        trace_index.route = trace_index.route.fillna(-1)
        nacnt = -1
        for i, g in trace_index.groupby([(trace_index.route != trace_index.route.shift()).cumsum()]):
            if g.route.mean() == nacnt:
                nacnt -= 1
                continue
            if nacnt < g.route.mean() < 0:
                trace_index.loc[g.index, 'route'] = nacnt
                nacnt -= 1

        # results are consistent with github ticket. number of route geometry points is the sum of number of tracepoints
        # plus the number of traversed nodes (total nodes minus first and last). Only inconsistency is the number of
        # geometry points obtained stitching all leg/step geometries (using the method below): it is a bit higher,
        # maybe because of precision... try with tolerance and/or comparing to matched coordinates plus looking up nodes
        return trace_index

    @property
    def matches(self):
        """

        :return:
        """
        return self._matches

    def plot(self, display=True):
        """

        :param display:
        :return:
        """
        trace = {k: [dic[k] for dic in self._trace] for k in self._trace[0]}
        data = list([
            go.Scattermapbox(
                mode="lines+markers",
                lon=trace['longitude'],
                lat=trace['latitude'],
                name='input trace'
            ),
            go.Scattermapbox(
                mode="lines+markers",
                lon=[tr.location[0] if tr is not None else None for tr in self._matches.tracepoints],
                lat=[tr.location[1] if tr is not None else None for tr in self._matches.tracepoints],
                name='output trace'
            )])
        for ixr, route in enumerate(self._matches.matchings):
            route_polyline = list(zip(*polyline.decode(route.geometry)))
            data.append(go.Scattermapbox(
                mode="markers",
                lon=route_polyline[1],
                lat=route_polyline[0],
                name='route ' + str(ixr) + ' geometry'
            ))
            route_legs_polyline = []
            for ixl, leg in enumerate(route.legs):
                # nodes_coordinates = leg.annotation.nodes
                # data.append(go.Scattermapbox(
                #     mode="lines+markers",
                #     lon=self._matches.matchings,
                #     lat=self._matches.matchings,
                #     name='matched coordinates'
                # ))
                for ixs, step in enumerate(leg.steps):
                    route_legs_polyline.append(list(zip(*polyline.decode(step.geometry))))
            # data.append(go.Scattermapbox(
            #     mode="markers",
            #     lon=route_legs_polyline[1],
            #     lat=route_legs_polyline[0],
            #     name='route legs/steps geometry'
            # ))
            # data.append(go.Scattermapbox(
            #     mode="markers",
            #     lon=[i.location[0] for i in step.intersections],
            #     lat=[i.location[1] for i in step.intersections],
            #     name='step ' + str(ixs) + ' intersections'
            # ))
            # data.append(go.Scattermapbox(
            #     mode="markers",
            #     lon=[m.location[0] for m in step.maneuvers],
            #     lat=[m.location[1] for m in step.maneuvers],
            #     name='step ' + str(ixs) + ' maneuvers'
            # ))
        layout = go.Layout(
            showlegend=True,
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(
                    lat=np.mean(trace['latitude']),
                    lon=np.mean(trace['longitude'])
                )
            )
        )
        if display:
            fig = go.Figure(data=data, layout=layout)
            fig.show()
        else:
            return data, layout


class NearestService:

    def __init__(self, config=None):
        if config is not None:
            self._config = config
        else:
            self._config = Config()

    def _request(self, lon, lat):
        url = "http://{}:{}/nearest/v1/driving/{},{}?".format(self._config.osrm['host'], self._config.osrm['port'], lon,
                                                              lat)
        params = {'number': 1}
        response = requests.get(url, params)
        return response.json()


class RouteService:

    def __init__(self, config=None):
        if config is not None:
            self._config = config
        else:
            self._config = Config()

    def _request(self, lon1, lat1, lon2, lat2):
        url = "http://{}:{}/route/v1/driving/{},{};{},{}?".format(self._config.osrm['host'], self._config.osrm['port'],
                                                                  lon1, lat1, lon2, lat2)
        params = {
            'steps': 'true',
            'geometries': 'polyline',
            'overview': 'full',
            'annotations': 'true'
        }
        response = requests.get(url, params)
        return RouteResult(**response.json())


class TableService:

    def __init__(self, config=None):
        if config is not None:
            self._config = config
        else:
            self._config = Config()

    def _request(self, slat, slon, dlatlist, dlonlist):
        sstring = "{},{}".format(slon, slat)
        astring = functools.reduce(lambda x, y: x + ";{},{}".format(dlonlist[y], dlatlist[y]), range(len(dlatlist)),
                                   sstring)
        url = "http://{}:{}/table/v1/driving/{}?".format(self._config.osrm['host'], self._config.osrm['port'], astring)
        params = {
            'sources': 0,
            'annotations': 'distance,duration'
        }
        response = requests.get(url, params)
        return response.json()

