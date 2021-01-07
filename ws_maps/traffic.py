#  Copyright (c) 2020. AV Connect Inc.
"""This module provides access to traffic data. """

import ws_maps.models as models
import ws_maps.network as network
from ws_maps.config import Config
from ws_maps.here import TrafficFlow
from ws_maps.tmc import Tmc
from ws_maps.network import BBID
from ws_maps.s3 import Blob
import sys
import daemon
import logging
import datetime
import dateutil
import time
import json
import pandas as pd
from sqlalchemy import Column, DateTime, String, Integer
from sqlalchemy.ext.declarative import declarative_base
import time

config = Config()
r = config.redis
Base = declarative_base()


class TrafficClient:
    """This class provides methods to get traffic data for a route or a network, at given times

    """

    def __init__(self):
        # start_time = time.time()
        self._config = Config()
        self._prefix = "traffic_"
        self._s3_blobs = Blob()
        # print("traffic client init took {} seconds ---".format((time .time()- start_time)))

    def at(self, locations=None, date_time='latest'):
        """Returns known traffic flow information, at the requested location and for the requested date and time.

        Location is a route or a network.
        Date and time can be specified with a datetime object for past dates and times, in which case the traffic
        information with the closest datetime will be returned. Alternatively, users can request the latest known
        traffic information, or to download a new one.

        :param locations:
        :type locations:
        :param date_time: either a past datetime, or a string set to 'latest' or 'new'
        :type date_time:
        :return:
        """
        start_time = time.time()

        assert (locations is not None) and hasattr(locations, 'edges') and isinstance(locations.edges, pd.DataFrame)

        if isinstance(date_time, str):
            assert date_time in ['latest', 'new']
        elif isinstance(date_time, list):
            assert all(l is None or isinstance(l, datetime.datetime) for l in date_time)
            assert(len(date_time) == len(locations.edges))
        else:
            assert isinstance(date_time, datetime.datetime)

        traffic_id = None
        if date_time == 'latest':
            timestamp = self._config.redis.get("timestamp_" + locations.bbid)
            if timestamp is not None:
                traffic_id = "traffic_" + timestamp + "_" + locations.bbid
            # else fixme
        elif isinstance(date_time, datetime.datetime):
            traffic_id = 'traffic_' + str(date_time) + '_' + str(locations.bbid)

        # id_time = time.time()
        # print("traffic client/at: getting traffic id took {} seconds ---".format((id_time - start_time)))

        if isinstance(date_time, list):
            # associate edges to different traffic blobs based on provided date_time
            edges = locations.edges
            edges['datetime'] = date_time
            edges_traffic = None

            for _, group in edges.groupby('datetime', dropna=False):

                if group.datetime.isnull().any():
                    group_traffic = pd.DataFrame(
                        columns=['traffic_speed_capped', 'traffic_speed_uncapped', 'free_flow_speed', 'jam_factor',
                                 'confidence', 'order'],
                        index=group.index)
                    group_traffic['order'] = group.order

                else:

                    if locations.bbid == 'chino':  # fixme patch for inconsistent chino bbid
                        traffic_id = 'traffic_' + str(group.datetime.iloc[0]) + '_Chino'
                    else:
                        traffic_id = 'traffic_' + str(group.datetime.iloc[0]) + '_' + str(locations.bbid)

                    traffic_blob = TrafficFlow(json=self._get_blob(traffic_id)).json
                    tmc_table = Tmc(bbid=locations.bbid, traffic_blob=traffic_blob)
                    group_traffic = tmc_table.traffic(group)

                if edges_traffic is None:
                    edges_traffic = group_traffic
                else:
                    edges_traffic = edges_traffic.append(group_traffic)

            return edges_traffic.sort_values(by=['order'])

        elif date_time is 'new' or traffic_id is None:
            bbid = models.get_session().query(network.BBID).filter(network.BBID.bbid == locations.bbid).all()[0]
            traffic_blob = TrafficFlow().get(bbid.min_lat, bbid.min_lon, bbid.max_lat, bbid.max_lon)

        else:
            traffic_blob = TrafficFlow(json=self._get_blob(traffic_id)).json

        # blob_time = time.time()
        # print("traffic client/at: getting traffic blob took {} seconds ---".format((blob_time - id_time)))
        tmc_table = Tmc(bbid=locations.bbid, traffic_blob=traffic_blob)
        # tmc_time = time.time()
        # print("traffic client/at: getting tmc table took {} seconds ---".format((tmc_time - blob_time)))
        edges_traffic = tmc_table.traffic(locations.edges)
        # edges_time = time.time()
        # print("traffic client/at: getting edges traffic took {} seconds ---".format((edges_time - tmc_time)))
        return pd.concat([locations.edges.copy(), edges_traffic], axis=1)

    def now(self, bbid):
        """

        :param bbid:
        :return:
        """
        if not isinstance(bbid, network.BBID):
            if isinstance(bbid, str):
                bbid = models.get_session().query(network.BBID).filter(network.BBID.bbid == bbid).all()[0]
            else:
                raise BaseException()  # fixme

        traffic_flow = TrafficFlow()
        json_blob = traffic_flow.get(bbid.min_lat, bbid.min_lon, bbid.max_lat, bbid.max_lon)

        timestamp = str(datetime.datetime.now())
        traffic_id = 'traffic_' + timestamp + '_' + str(bbid)

        if 'delete_old_traffic' in self._config.config.keys() and self._config.config['delete_old_traffic']:
            self._delete_old(bbid)

        if 'persist_blob' in self._config.config.keys() and self._config.config['persist_blob']:
            self._set_blob(traffic_id, None, json_blob)

        self._config.redis.set(traffic_id + '_raw', json.dumps(json_blob))
        self._config.redis.set('timestamp_' + bbid, timestamp)

        return json_blob

    def get(self, traffic_id):
        """

        :param traffic_id:
        :return:
        """
        return self._get_blob(traffic_id)

    def _get_blob(self, traffic_id):
        """

        :param traffic_id:
        :type traffic_id:
        :return:
        :rtype:
        """
        traffic = models.get_session().query(Traffic).filter(
            Traffic.mapped_traffic_key == traffic_id).all()[0]
        # flow = self._s3_blobs.get(traffic.mapped_traffic_key)
        raw = self._s3_blobs.get(traffic.raw_traffic_key)
        return raw

    def _set_blob(self, traffic_id, traffic_flow, traffic_raw):
        """Persists traffic flow data to S3 and makes a record in postgres and redis

        :param traffic_id:
        :type traffic_id:
        :param traffic_flow:
        :type traffic_flow:
        :param traffic_raw:
        :type traffic_raw:
        :return:
        :rtype:
        """
        traffic = Traffic()
        tokens = traffic_id.split("_")
        traffic.bbid = tokens[2]
        traffic.date_measured = dateutil.parser.parse(tokens[1])
        traffic.raw_traffic_key = traffic_id + '_raw'
        traffic.mapped_traffic_key = traffic_id
        self._s3_blobs.set(traffic.mapped_traffic_key, traffic_flow)
        self._s3_blobs.set(traffic.raw_traffic_key, traffic_raw)
        models.get_session().add(traffic)
        models.get_session().commit()

    def list_blobs(self):
        """

        :return:
        :rtype:
        """
        return [key for key in self._s3_blobs.list_keys(self._prefix)]

    def _delete_old(self, bbid):
        """Remove traffic records

        :param bbid:
        :type bbid:
        :return:
        :rtype:
        """
        try:
            traffic_timestamp = self._config.redis.get('timestamp_' + bbid)
            self._config.redis.delete('traffic_' + traffic_timestamp + '_' + str(bbid))
            self._config.redis.delete('traffic_' + traffic_timestamp + '_' + str(bbid) + '_raw')
        except BaseException as e:
            print("error deleting traffic from " + bbid + " " + str(e.__class__))

    def plot(self):
        """

        :return:
        """
        pass

    # def _find_blob(self, date_time):
    #     """
    #
    #     :param date_time:
    #     :type date_time:
    #     :return:
    #     :rtype:
    #     """
    #     persisted_blobs = self.list_blobs()
    #     date_time_list = []  # get from s3 or postgres?? update postgres then filter only this bbid
    #     closest_date_time = min(date_time_list, key=lambda d: abs(d - date_time))


class Traffic(models.Base, models.ConnectedModel):
    """A traffic blob recorded at a certain date and time for a certain BBID. """
    __tablename__ = 'traffic'
    id = Column(Integer, primary_key=True)
    date_measured = Column(DateTime, nullable=False, index=True)
    bbid = Column(String(250), nullable=False)
    mapped_traffic_key = Column(String(100))
    raw_traffic_key = Column(String(100))

    def _get_blob_name(self, json=True):
        """ """
        if json:
            return 'traffic_' + str(self.date_measured) + '_' + self.bbid
        else:
            return 'traffic_' + str(self.date_measured) + '_' + self.bbid + '_raw'

    def get_data(self):
        """ Get traffic blob.

        :return: traffic blob
        """
        flow, raw = TrafficClient().get(self.mapped_traffic_key)
        return flow, raw

    def set_data(self, data, json=True):
        """Push traffic blob into redis

        :param data:
        :param json:
        """
        r.set(self._get_blob_name(json), data)

    @classmethod
    def traffic_from_s3(cls):
        """ """
        traffic_list = TrafficClient().list_blobs()
        tl = len(traffic_list)
        print("traffic list loaded")
        for ix, traffic_blob in enumerate(traffic_list):
            print("{} of {}".format(ix, tl))
            try:
                tokens = traffic_blob.split("_")
                if tokens[-1] == 'raw' and tokens[-2] == 'Chino':
                    found_traffic = models.session.query(Traffic).filter(Traffic.raw_traffic_key == traffic_blob).all()
                    if len(found_traffic) == 0:
                        found_bbid = models.session.query(BBID).filter(BBID.bbid == 'chino').all()
                        if len(found_bbid) > 0:
                            bbid = found_bbid[0]
                            t = Traffic()
                            t.bbid = bbid.bbid
                            t.date_measured = dateutil.parser.parse(tokens[1])
                            t.raw_traffic_key = traffic_blob
                            t.mapped_traffic_key = traffic_blob[:-4]
                            models.session.add(t)
                            models.session.commit()
                            print("committed bbid {} date {}".format(t.bbid, str(t.date_measured)))
            except BaseException as e:
                print(e.__class__.__name__)
                print(e.message)
                models.session.rollback()


class TrafficDaemon:
    """ """

    def __init__(self, config=None):
        if config is not None and isinstance(config, Config):
            self._config = config
        else:
            self._config = Config()
        self._traffic = TrafficClient()
        self._setup_logging()

    def _setup_logging(self):
        """

        """
        format = "%(asctime)-15s %(message)s"
        if "logging_dir" in self._config.config.keys():  # fixme do this in config.py
            logging_dir = self._config.config['logging_dir']
        else:
            logging_dir = self._config._logging_dir
        logging.basicConfig(format=format,
                            filename=logging_dir + 'traffic.log',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def spin(self):
        """Get the latest traffic and match it to arcs for each bbid

        :return:
        """
        # fixme
        for bbid in models.get_session().query(models.BBID).filter(models.BBID.bbid == 'chino').all():  # fixme
            try:
                ret = self._traffic.pull(bbid, 'new')
                logging.info('Finish getting traffic')
            except BaseException as e:
                logging.exception("Error getting traffic " + str(e))
            logging.info("done")

    def status(self):
        """

        :return:
        """
        pass


def main(config):
    """

    :param config:
    :type config:
    """
    traffic_daemon = TrafficDaemon(config)
    logging.info("starting traffic daemon")
    while True:
        traffic_daemon.spin()
        logging.info("ran traffic request")
        time.sleep(config.config['traffic_interval'])


if __name__ == "__main__":
    cfg = Config()
    if "-d" in sys.argv:
        with daemon.DaemonContext(working_directory=cfg._src_dir):
            main(cfg)
    else:
        main(cfg)

