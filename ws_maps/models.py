#  Copyright (c) 2020. AV Connect Inc.
""" """

from ws_maps.config import Config
# from ws_maps.network import Network
# import ws_maps.traffic as traffic
import datetime
import dateutil
import pytz
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, Float, BigInteger, create_engine
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import subprocess
import os


config = Config()
r = config.redis
Base = declarative_base()


def db_init():
    """

    """
    my_env = os.environ.copy()
    my_env['TESTING'] = 'true'
    subprocess.call(['alembic', 'downgrade', 'base'], env=my_env)
    subprocess.call(['alembic', 'upgrade', 'head'], env=my_env)


def init_engine(config):
    """

    :param config:
    :return:
    """
    engine = create_engine(config.db_uri)
    Base.metadata.bind = engine
    DBSession = sessionmaker()
    DBSession.bind = engine
    session = scoped_session(DBSession)
    return engine, session


engine, session = init_engine(config)


def get_session():
    """

    :return:
    """
    return session


def get_engine():
    """

    :return:
    """
    return engine


class ConnectedModel(object):
    """The ConnectedModel class mixes in convenience methods for sqlalchemy objects using the base db connection. """

    def create(self):
        """Creates a new instance. """
        session.add(self)
        session.commit()

    @classmethod
    def all(cls):
        """

        :return:
        """
        return session.query(cls).all()

    @classmethod
    def get(cls, id):
        """

        :param id:
        :return:
        """
        return session.query(cls).get(id)

    @classmethod
    def filter(cls, filter_string):
        """

        :param filter_string:
        :return:
        """
        return session.query(cls).filter(filter_string).all()

    def to_dict(self):
        """

        :return:
        """
        types = (int, float, datetime.datetime, BigInteger)
        me = self.__dict__
        return {key: me[key] for key in me if not me[key] or type(me[key]) in types}

    @classmethod
    def as_utc(cls, field):
        """

        :param field:
        :return:
        """
        return pytz.utc.localize(field)

