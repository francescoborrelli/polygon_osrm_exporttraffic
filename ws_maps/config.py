#  Copyright (c) 2020. AV Connect Inc.
"""This module configures the parameters used in the other modules. """

import os
import influxdb
import redis
import yaml
from pathlib import Path
import re


class PostgresConn:

    def __init__(self, uri):
        conn = re.split('://|:|@|/', uri)
        self.uri = uri
        self.user = conn[1]
        self.password = conn[2]
        self.host = conn[3]
        self.port = conn[4]
        self.database = conn[5]


class Config:
    """

    """
    def __init__(self):
        self._config = self._get_config()
        self._influx = influxdb.client.InfluxDBClient(
            self._config['influx']['host'],
            self._config['influx']['port'],
            database=self._config['influx']['db'])
        self._influx_pandas = influxdb.dataframe_client.DataFrameClient(
            self._config['influx']['host'],
            self._config['influx']['port'],
            database=self._config['influx']['db'])
        self._redis = redis.Redis(
            self._config['redis']['host'],
            self._config['redis']['port'])
        self._gisdb = PostgresConn(self.config['gisdb']['uri'])
        self._routingdb = PostgresConn(self.config['pgrouting']['uri'])
        self._db = PostgresConn(self.config['db']['uri'])
        self._project_dir = str(Path(__file__).parents[1])
        self._src_dir = str(Path(__file__).parents[0])
        self._logging_dir = self._project_dir + '/logs/'

    @property
    def config(self):
        """

        :return:
        """
        return self._config

    @property
    def db(self):
        """

        :return:
        """
        return self._db

    @property
    def influx(self):
        """

        :return:
        """
        return self._influx

    @property
    def influx_pandas(self):
        """

        :return:
        """
        return self._influx_pandas

    @property
    def redis(self):
        """

        :return:
        """
        return self._redis

    @property
    def gisdb(self):
        """

        :return:
        """
        return self._gisdb

    @property
    def routingdb(self):
        """

        :return:
        """
        return self._routingdb

    @property
    def osrm(self):
        """

        :return:
        """
        return self._config['osrm']

    @property
    def mapbox(self):
        """

        :return:
        """
        return self._config['mapbox']

    @property
    def open_elevation(self):
        """

        :return:
        """
        return self._config['open_elevation']

    @property
    def open_weather(self):
        """

        :return:
        """
        return self._config['open_weather']

    @property
    def here(self):
        """

        :return:
        """
        return self._config['here']

    @property
    def s3_bucket(self):
        """

        :return:
        """
        return self._config['s3_bucket']

    @property
    def pgrouting(self):
        """

        :return:
        """
        return self._config['pgrouting']

    @classmethod
    def _get_config(cls):
        """

        :return:
        """
        try:
            directory = os.path.dirname(os.path.realpath(__file__))
            with open(directory + "/settings.yaml", "r") as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        except BaseException as e:
            # print(e)
            # print("Looking for settings in root directory trying root directory ")
            with open(os.environ.get("ROOT") + "/settings.yaml", "r") as f:
                return yaml.load(f, Loader=yaml.SafeLoader)

