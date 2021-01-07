# from ws_maps.pgrouting import PGRouting
import pandas as pd
import geopandas as gpd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, load_only
import sqlalchemy as sa
from sqlalchemy import Column, String, Integer, Float, BigInteger, ARRAY
from geoalchemy2 import Geometry


engine_osm = sa.create_engine('postgresql://postgres:postgres@localhost:5439/gisdb2')
BaseOsm = declarative_base()
BaseOsm.metadata.bind = engine_osm
osmdb_session = sessionmaker()
osmdb_session.bind = engine_osm
session_osm = scoped_session(osmdb_session)


class Way(BaseOsm):
    """Models an OpenStreetMap way. """

    __tablename__ = 'planet_osm_ways'
    id = Column(BigInteger, primary_key=True)
    nodes = Column(ARRAY(BigInteger), nullable=False)
    tags = Column(String, nullable=True)


# engine_osm2po = sa.create_engine('postgresql://postgres:postgres@localhost:5432/routing_osm2po')
engine_osm2po = sa.create_engine('postgresql://postgres:postgres@localhost:5439/osm2podb')
BaseOsm2po = declarative_base()
BaseOsm2po.metadata.bind = engine_osm2po
osm2podb_session = sessionmaker()
osm2podb_session.bind = engine_osm2po
session_osm2po = scoped_session(osm2podb_session)


class Edge(BaseOsm2po):
    __tablename__ = 'osm_2po_4pgr'
    id = Column(Integer, primary_key=True, nullable=False)
    osm_id = Column(BigInteger)
    osm_name = Column(String)
    osm_meta = Column(String)
    osm_source_id = Column(BigInteger)
    osm_target_id = Column(BigInteger)
    clazz = Column(Integer)
    flags = Column(Integer)
    source = Column(Integer, index=True)
    target = Column(Integer, index=True)
    km = Column(Float)
    kmh = Column(Integer)
    cost = Column(Float)
    reverse_cost = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    geom_way = Column(Geometry)
    the_geom_line = Column(Geometry)
    nodes = Column(ARRAY(BigInteger))


def get_sublist(list, start, end):
    sublist = []
    for element in list:
        if len(sublist) == 0:
            if element == start:
                sublist = [element]
        else:
            sublist.append(element)
            if element == end:
                break
    return sublist


if __name__ == '__main__':

    # print("dumping osm2po edges table to csv")
    # pgr = PGRouting(user='postgres', password='postgres', host='localhost', port=5439, database='osm2podb')
    # pgr.edges_to_csv('/home/jacopo/edges_dump_test.csv', ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id',
    # 'cost', 'reverse_cost'])

    print("getting osm2po edges table")
    columns = ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id', 'cost', 'reverse_cost', 'geom_way']
    edges = gpd.read_postgis(sql=session_osm2po.query(Edge).options(load_only(*columns)).statement,
                             con=engine_osm2po, geom_col='geom_way')

    print("getting osm ways table")
    columns = ['nodes']
    ways_nodes = pd.read_sql(
        sql=session_osm.query(Way).options(load_only(*columns)).filter(Way.id.in_(edges.osm_id)).statement,
        con=engine_osm).set_index('id')

    def get_nodes(row):
        try:
            return get_sublist(ways_nodes.loc[row.osm_id, 'nodes'], row.osm_source_id, row.osm_target_id)
        except KeyError:
            return []

    print("adding nodes column to the edges table")
    edges['nodes'] = edges.apply(get_nodes, axis=1)
    # edges['nodes'] = edges.apply(lambda x: get_sublist(ways_nodes.loc[x.osm_id, 'nodes'],
    #                                                    x.osm_source_id, x.osm_target_id), axis=1)

    print("checking that the node lists added to the edges table are consistent with the geometries")
    edges['nodes_match_geom'] = edges.apply(lambda x: len(x.nodes) == sum([len(g.coords) for g in x.geom_way]), axis=1)
    print("the node list is consistent with the geometry in {} edges out of {}".format(edges.nodes_match_geom.sum(),
                                                                                       len(edges)))

    # edges.to_csv('/home/jacopo/edges_nodes.csv')

    # stmt = Edge.update().values(nodes=edges.nodes).where(Edge.osm_id == edges.osm_id)
    # session_osm2po.query(Edge).filter(Edge.osm_id == edges.osm_id.to_list()).update({"nodes": (edges.nodes.to_list())})
    # session_osm2po.commit()
    # edges.to_postgis('osm_2po_4pgr', engine_osm2po)                                                                                       len(edges)))
    edges.to_csv('all_osm_edges.csv')