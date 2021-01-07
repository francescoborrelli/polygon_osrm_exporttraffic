#  Copyright (c) 2020. AV Connect Inc.
"""This module enables requests to Here Maps APIs and converts the responses to formats that can be consumed by other
ws_maps modules. """

from ws_maps.config import Config
import requests
from shapely.geometry import LineString
import shapely.wkt
import shapely.ops
import pandas as pd
import geopandas as gpd


class TrafficFlow:
    """Pulls traffic flow data from Here maps.

    :param json:
    :type json:
    """

    def __init__(self, json=None):
        self._config = Config()
        self._json = None
        self._df = None
        self._gdf = None

        if json is not None:
            self._json = json
            self._df = TrafficFlow.json_to_df(json)
            self._gdf = TrafficFlow.json_to_gdf(json)

    @property
    def json(self):
        return self._json

    @property
    def df(self):
        if self._df is not None:
            return self._df
        elif self._json is not None:
            return TrafficFlow.json_to_df(self._json)
        else:
            return None

    @property
    def gdf(self):
        if self._gdf is not None:
            return self._df
        elif self._json is not None:
            return TrafficFlow.json_to_gdf(self._json)
        else:
            return None

    def get(self, lat_min, lon_min, lat_max, lon_max):
        """

        :param lat_min:
        :type lat_min:
        :param lat_max:
        :type lat_max:
        :param lon_min:
        :type lon_min:
        :param lon_max:
        :type lon_max:
        :return:
        :rtype:
        """

        response = requests.get("https://traffic.cit.api.here.com/traffic/6.3/flow.json?" +
                                "app_id=" + self._config.here['app_id'] +
                                "&app_code=" + self._config.here['app_code'] +
                                "&bbox={},{};{},{}".format(lat_max, lon_min, lat_min, lon_max) +
                                "&units=metric" +
                                "&responseattributes=sh"
                                )
        self._json = response.json()
        return self._json

    @staticmethod
    def json_to_list(json):
        """Converts a traffic blob from Here's Traffic Flow API to a gpd.GeoDataFrame

        :param json: traffic blob from Here's Traffic Flow API
        :type json: dict
        :return: a GeoDataFrame with a traffic flow item per row
        :rtype: gpd.GeoDataFrame
        """

        # parse traffic blob
        flow_items_list = []
        for rws_ix, rws in enumerate(json['RWS']):  # RWS = roadway list
            for rw_ix, rw in enumerate(rws['RW']):  # RW = roadway
                for fis_ix, fis in enumerate(rw['FIS']):  # FIS = flow item list
                    for fi_ix, fi in enumerate(fis['FI']):  # FI = flow item
                        shp_wkt = 'MULTILINESTRING('
                        for shp in fi['SHP']:  # SHP = shapefile
                            for line in shp['value']:
                                if line[-1] == ' ':
                                    line = line[:-1]
                                shp_wkt += '(' + line.translate(str.maketrans(', ', ' ,')) + '),'
                        shp_wkt = shp_wkt[:-1] + ')'
                        flow_item = {
                            'RWS': rws_ix,
                            'RW': rw_ix,
                            'PBT': rw['PBT'] if 'PBT' in rw.keys() else None,
                            'DE_RW': rw['DE'] if 'DE' in rw.keys() else None,
                            'mid': rw['mid'] if 'mid' in rw.keys() else None,
                            'LI': rw['LI'] if 'LI' in rw.keys() else None,
                            'FIS': fis_ix,
                            'FI': fi_ix,
                            # 'TMC': fi['TMC'],  # An ordered collection of TMC locations
                            'PC': fi['TMC']['PC'],
                            'LE': fi['TMC']['LE'],
                            'DE': fi['TMC']['DE'],
                            'QD': fi['TMC']['QD'],
                            # 'CF': fi['CF'],  # Current Flow. Details about speed and Jam Factor information
                            'CN': fi['CF'][0]['CN'] if 'CN' in fi['CF'][0].keys() else None,
                            'TY': fi['CF'][0]['TY'] if 'TY' in fi['CF'][0].keys() else None,
                            'SP': fi['CF'][0]['SP'] if 'SP' in fi['CF'][0].keys() else None,
                            'SU': fi['CF'][0]['SU'] if 'SU' in fi['CF'][0].keys() else None,
                            'FF': fi['CF'][0]['FF'] if 'FF' in fi['CF'][0].keys() else None,
                            'JF': fi['CF'][0]['JF'] if 'JF' in fi['CF'][0].keys() else None,
                            'CF_extras': (len(fi['CF']) > 1),  # is there further info that we currently don't parse?
                            # 'SHP': MultiLineString(line_string_list[:])}  # Shapefile, converted to a shapely object
                            'SHP': shp_wkt}
                        flow_items_list.append(flow_item)
        return flow_items_list

    @staticmethod
    def json_to_df(json):
        return pd.DataFrame(TrafficFlow.json_to_list(json))

    @staticmethod
    def df_to_gdf(df):
        geometry = df['SHP'].map(shapely.wkt.loads).apply(shapely.ops.linemerge)
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
        # fixme deal with the ones that have to be multistring (duplicate row but they can't be matched in one shot)
        gdf_ = gdf[gdf.geometry.apply(lambda x: isinstance(x, LineString))]
        # todo need? keep as check?
        # gdf_.groupby(['DE_RW', 'QD']).apply(
        #     lambda x: all([g0.touches(g1) for g0, g1 in pairwise(x.geometry.tolist())])).sum()
        return gdf_

    @staticmethod
    def json_to_gdf(json):
        df = TrafficFlow.json_to_df(json)
        return TrafficFlow.df_to_gdf(df)


class Waypoint:
    """

    :ivar linkId:
    :ivar mappedPosition:
    :ivar originalPosition:
    :ivar type:
    :ivar spot:
    :ivar sideOfStreet:
    :ivar mappedRoadName:
    :ivar label:
    :ivar shapeIndex:
    :ivar source:
    """
    _fields = []


class Link:
    """

    :ivar linkId:
    :ivar shape:
    :ivar firstPoint:
    :ivar lastPoint:
    :ivar length:
    :ivar remainDistance:
    :ivar remainTime:
    :ivar nextLink:
    :ivar maneuver:
    :ivar dynamicSpeedInfo:
    :ivar flags:
    :ivar functionalClass:
    :ivar roadNumber:
    :ivar timezone:
    :ivar truckRestrictions:
    :ivar roadName:
    :ivar consumption:
    """
    _fields = []


class Leg:
    """Represents a route through two or more waypoints.

    :ivar start:
    :ivar end:
    :ivar length:
    :ivar travelTime:
    :ivar maneuver:
    :ivar turnByTurnManeuver:
    :ivar link:
    :ivar boundingBox:
    :ivar shape:
    :ivar firstPoint:
    :ivar lastPoint:
    :ivar trafficTime:
    :ivar baseTime:
    :ivar stayingTime:
    :ivar summary:
    """
    _fields = []


class Route:
    """Represents a route through two or more waypoints.

    :ivar waypoint:
    :ivar mode:
    :ivar shape:
    :ivar boundingBox:
    :ivar leg:
    :ivar note:
    :ivar summary:
    :ivar maneuverGroup:
    :ivar label:
    :ivar zone:
    """
    _fields = []


class Summary:
    """

    :ivar distance:
    :ivar flags:
    :ivar text:
    :ivar base_time:
    :ivar traffic_time:
    :ivar travel_time:
    """
    _fields = []

