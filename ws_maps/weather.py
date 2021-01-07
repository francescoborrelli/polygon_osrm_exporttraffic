#  Copyright (c) 2020. AV Connect Inc.
"""This module provides access to weather data. """

from ws_maps.config import Config
import requests
import pandas as pd
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from shapely.ops import cascaded_union, split
from shapely.geometry import Point, Polygon, LineString
import math


class Weather:
    """

    """

    # todo implement blob persistence and network -> weather locations map persistence
    # todo implement historical/forecast data pull
    # todo add bbid as input and get bbox from there

    def __init__(self):
        self._config = Config()
        self._base_url = "http://api.openweathermap.org/data/2.5/"
        self._zoom_level = 15

    def _pull_live_bbox(self, lat_min, lon_min, lat_max, lon_max):
        bbox = Polygon([[lon_min, lat_min], [lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min]]).buffer(0.000001)
        url = self._base_url + "box/city?bbox={},{},{},{},{}&units=metric&appid=".format(
            lon_min, lat_min, lon_max, lat_max, self._zoom_level) + self._config.open_weather['app_id']
        response = requests.get(url)
        if response.ok:
            json_response = response.json()
            if not json_response:
                # list is empty if the bbox is too small; in this case, request the city in which any of the points lies
                url = self._base_url + "find?lat={}&lon={}&cnt={}&units=metric&appid=".format(
                          (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0, 1) + self._config.open_weather['app_id']
                json_response = requests.get(url).json()
                # todo check response ok
            df = pd.DataFrame(json_response['list'])
            df = pd.concat([df.drop(['coord'], axis=1), df['coord'].apply(pd.Series)], axis=1)
            if 'Lat' in df.columns:
                df['geometry'] = df.apply(lambda x: Point(x.Lon, x.Lat), axis=1)
            else:
                df['geometry'] = df.apply(lambda x: Point(x.lon, x.lat), axis=1)
            df = pd.concat([df.drop(['main'], axis=1), df['main'].apply(pd.Series)], axis=1)
            df = pd.concat([df.drop(['wind'], axis=1), df['wind'].apply(pd.Series)], axis=1)
            return df, json_response, bbox
        else:
            raise BaseException("Error getting weather {} {}".format(str(response.status_code), response.content))

    def _make_bbox_lookup(self, bbox, locations):
        """Make Voronoi partition of bbox based on received list of locations.

        """

        bbox = gpd.GeoDataFrame(pd.DataFrame(columns=['geometry'], data=[bbox]), geometry='geometry')
        bbox_proj = bbox.set_crs(epsg=4326).to_crs(epsg=3395)  # Web Mercator projection
        boundary_shape = cascaded_union(bbox_proj.geometry)

        locations = gpd.GeoDataFrame(locations, geometry='geometry')  # cities locations
        locations_proj = locations.set_crs(epsg=4326).to_crs(epsg=3395)  # Web Mercator projection
        coords = points_to_coords(locations_proj.geometry)

        if len(locations) > 2:
            poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)
            bbox_lookup = pd.DataFrame(columns=['locations'], data=pts)
            bbox_lookup['polygons'] = None
            for pt_list, poly in zip(poly_to_pt_assignments, poly_shapes):
                for pt in pt_list:
                    bbox_lookup.at[pt, 'polygons'] = poly
            return bbox_lookup
        elif len(locations) == 2:
            bbox_lookup = pd.DataFrame(columns=['locations', 'polygons'])
            bbox_lookup.locations = locations_proj.geometry.values
            minx, miny, maxx, maxy = bbox_proj.geometry.values[0].bounds
            bisector_length = math.hypot(maxx - minx, maxy - miny)
            connector = LineString(coords)
            bisector = LineString([connector.parallel_offset(bisector_length / 2, 'left').centroid,
                                   connector.parallel_offset(bisector_length / 2, 'right').centroid])
            bbox_lookup.polygons = split(bbox_proj.geometry.values[0], bisector)
            return bbox_lookup
        else:
            bbox_lookup = pd.DataFrame(columns=['locations', 'polygons'])
            bbox_lookup.locations = locations_proj.geometry.values
            bbox_lookup.polygons = bbox_proj.geometry.values
            return bbox_lookup

    def _lookup_live_coordinates(self, lat, lon):
        url = self._base_url + "weather?lat={}&lon={}&units=metric&appid=".format(lat, lon) +\
              self._config.open_weather['app_id']
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            raise BaseException("Error getting weather {} {}".format(str(response.status_code), response.content))

    def at(self, locations):

        if (hasattr(locations, 'nodes') and isinstance(locations.nodes, pd.DataFrame) and hasattr(locations, 'edges')
                and isinstance(locations.edges, pd.DataFrame)):
            lat_min = locations.nodes.latitude.min()
            lat_max = locations.nodes.latitude.max()
            lon_min = locations.nodes.longitude.min()
            lon_max = locations.nodes.longitude.max()
            df, json_response, bbox = self._pull_live_bbox(lat_min, lon_min, lat_max, lon_max)
            lookup = self._make_bbox_lookup(bbox, df.geometry)
            query_points = locations.nodes.copy()
            query_points['geometry'] = query_points.apply(lambda x: Point(x.longitude, x.latitude), axis=1)
            query_points = gpd.GeoDataFrame(query_points, geometry='geometry')
            query_points_proj = query_points.set_crs(epsg=4326).to_crs(epsg=3395)

            def find_polygon(x):
                for ix, row in lookup.iterrows():
                    if x.geometry.within(row.polygons):
                        return ix

            query_points_proj['polygons'] = query_points_proj.apply(find_polygon, axis=1)
            nodes = pd.concat([
                locations.nodes.copy().reset_index(),
                df.loc[query_points_proj.polygons.values, :].drop(columns=['id', 'geometry']).reset_index()],
                axis=1).set_index('id')
            nodes.temp = nodes.temp  # - 273.15
            nodes.temp_min = nodes.temp_min  # - 273.15
            nodes.temp_max = nodes.temp_max  # - 273.15
            nodes.feels_like = nodes.feels_like  # - 273.15

            edges = locations.edges.copy().reset_index()
            edges = pd.concat([edges,
                               edges.apply(lambda x: nodes.loc[[x.source, x.target], :]
                                           .mean(axis=0)
                                           .drop(['longitude', 'latitude', 'elevation']),
                                           axis=1)],
                              axis=1)
            edges = edges.set_index(['source', 'target'])
            return nodes, edges

        else:
            raise NotImplementedError


# class Weather(Base, ConnectedModel):
#     """A weather blob recorded at a certain date and time for a certain BBID. """
#     __tablename__ = 'weather'
#     id = Column(Integer, primary_key=True)
#     date_measured = Column(DateTime, nullable=False, index=True)
#     bbid = Column(String(250), nullable=False)

