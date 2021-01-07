#  Copyright (c) 2020. AV Connect Inc.
"""This module provides access to terrain elevation data. """

import requests
import pandas as pd
import numpy as np
from ws_maps.config import Config


class Terrain:
    """

    """

    def __init__(self):
        self._config = Config()
        self._base_url = "http://{}:{}/api/v1/lookup".format(self._config.open_elevation["host"],
                                                             self._config.open_elevation["port"])
        self._max_get_size = 90
        self._max_post_size = 500

    def _post(self, df):
        # POST request method - slightly less complicated than the GET method below, but fails on some requests that
        # instead work with the GET method; I suspect it is due to similar issues with the size of the request, which
        # would need to play with the chunk size (easy) and with the precision of lat/lon strings (control it in the
        # json encoding, probably not using the json argument)
        dfr = df.dropna().reset_index()
        dfr['elevation'] = None
        dfr['error'] = None
        for _, g in dfr.groupby(np.arange(len(dfr)) // self._max_post_size):
            response = requests.post(self._base_url,
                                     json={'locations': g[['latitude', 'longitude']].to_dict('records')})
            if response.ok:
                results = pd.DataFrame.from_records(response.json()['results'])
                if 'elevation' in results.columns:
                    dfr.loc[g.index, 'elevation'] = results.elevation.astype(float).values
                if 'error' in results.columns:
                    dfr.loc[g.index, 'error'] = results.error.values

        if 'id' in dfr.columns:
            dfr.set_index('id', inplace=True)
        return df.copy().join(dfr[['elevation', 'error']])

    def _get(self, df):
        # make a copy of the data without any nan values
        dfr = df.dropna().reset_index()
        dfr['elevation'] = None
        dfr['error'] = None

        # divide request in chunks; there is a 1024 bytes limit on the get request, so it is necessary to choose an
        # appropriate chunk size and to limit the number of decimal places in the lat/lon
        # IMPORTANT: the open-elevation server seems very prone to responding with inaccurate error codes; for instance
        # I had it complain about request formatting when making requests like below but just using chunks of 100 points
        for _, g in dfr.groupby(np.arange(len(dfr)) // self._max_get_size):
            url = self._base_url + "?locations="
            for ix, row in g.iterrows():
                url += "{:.6f},{:.6f}|".format(row.latitude, row.longitude)
            url = url[:-1]
            response = requests.get(url)
            if response.ok:
                results = pd.DataFrame.from_records(response.json()['results'])
                if 'elevation' in results.columns:
                    dfr.loc[g.index, 'elevation'] = results.elevation.astype(float).values
                if 'error' in results.columns:
                    dfr.loc[g.index, 'error'] = results.error.values
        if 'id' in dfr.columns:
            dfr.set_index('id', inplace=True)
        return df.copy().join(dfr[['elevation', 'error']])

    def at(self, locations=None, latitude=None, longitude=None):
        """

        :param locations:
        :type locations:
        :param latitude:
        :type latitude:
        :param longitude:
        :type longitude:
        :return:
        :rtype:
        """
        assert ((locations is not None) ^ ((latitude is not None) and (longitude is not None)))

        df = pd.DataFrame()
        if isinstance(locations, pd.DataFrame):
            assert 'latitude' in locations.columns
            assert 'longitude' in locations.columns
            df = locations.copy()
        elif hasattr(locations, 'nodes') and isinstance(locations.nodes, pd.DataFrame):
            df['latitude'] = locations.nodes.latitude
            df['longitude'] = locations.nodes.longitude
        else:
            try:
                df['latitude'] = latitude
                df['longitude'] = longitude
            except Exception:
                raise BaseException  # fixme

        if len(df) >= self._max_get_size:
            result = self._post(df)
        else:
            result = self._get(df)

        if hasattr(locations, 'edges') and isinstance(locations.edges, pd.DataFrame):
            edges = locations.edges.reset_index()
            edges['elevation'] = edges.apply(lambda x: result.loc[x.target, 'elevation'] -
                                                       result.loc[x.source, 'elevation'],
                                             axis=1)
            return result, edges.set_index(['source', 'target'])

        else:
            return result

