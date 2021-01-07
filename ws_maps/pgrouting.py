#  Copyright (c) 2020. AV Connect Inc.
"""This module provides a minimal interface to pgRouting via psycopg2. It has been adapted from psycopgr
(https://github.com/herrkaefer/psycopgr) to work with newer pgRouting versions, to work with our database, and to better
fit our needs in matching and routing. Notice that pgRouting has a much bigger API, this module only covers our current
needs but can be easily extended if necessary.
"""

from collections import namedtuple
from typing import List
import psycopg2
import psycopg2.extras
import psycopg2.sql
import pandas as pd
from typing import Union, Optional, Dict, List


PgrNode = namedtuple('PgrNode', ['id', 'osm_id', 'lon', 'lat'])


class PGRouting:
    """Wraps pgRouting shortest paths functions and edges and vertices tables queries.

    Computes shortest paths and costs from nodes to nodes, represented in geographic coordinates or node id, and queries
    existence and properties of nodes and edges by wrapping pgRouting.
    Initialization accepts the same database connection arguments that psycopg2 accepts.

    :param dbname: the database name (database is a deprecated alias)
    :param user: user name used to authenticate
    :param password: password used to authenticate
    :param host: database host address (defaults to UNIX socket if not provided)
    :param port: connection port number (defaults to 5432 if not provided)
    """
    # default edge table definition
    _meta_data = {
        'edges_table': psycopg2.sql.Identifier('osm_2po_4pgr'),
        'vertices_table': psycopg2.sql.Identifier('osm_2po_vertex'),
        'edges_id': psycopg2.sql.Identifier('id'),
        'edges_way_id': psycopg2.sql.Identifier('osm_id'),
        'edges_source': psycopg2.sql.Identifier('source'),
        'edges_target': psycopg2.sql.Identifier('target'),
        'edges_source_osm': psycopg2.sql.Identifier('osm_source_id'),
        'edges_target_osm': psycopg2.sql.Identifier('osm_target_id'),
        'edges_length': psycopg2.sql.Identifier('km'),
        'edges_cost': psycopg2.sql.Identifier('cost'),
        'edges_reverse_cost': psycopg2.sql.Identifier('reverse_cost'),
        'edges_x1': psycopg2.sql.Identifier('x1'),
        'edges_y1': psycopg2.sql.Identifier('y1'),
        'edges_x2': psycopg2.sql.Identifier('x2'),
        'edges_y2': psycopg2.sql.Identifier('y2'),
        'edges_geometry': psycopg2.sql.Identifier('geom_way'),
        'has_reverse_cost': psycopg2.sql.Literal(True),
        'directed': psycopg2.sql.Literal(False),
        'srid': psycopg2.sql.Literal(4326),
        'vertices_id': psycopg2.sql.Identifier('id'),
        'vertices_osm_id': psycopg2.sql.Identifier('osm_id'),
        'vertices_longitude': psycopg2.sql.Identifier('longitude'),
        'vertices_latitude': psycopg2.sql.Identifier('latitude'),
        'vertices_geometry': psycopg2.sql.Identifier('geom_vertex')
    }

    def __init__(self, *args, **kwargs):
        self._conn = None
        self._cur = None
        self._connect_to_db(*args, **kwargs)

    def __del__(self):
        self._close_db()

    def _connect_to_db(self, *args, **kwargs):
        if self._cur is not None and not self._cur.closed:
            self._cur.close()
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        try:
            self._conn = psycopg2.connect(*args, **kwargs)
            self._cur = self._conn.cursor(
                cursor_factory=psycopg2.extras.DictCursor)
        except psycopg2.Error as e:
            print(e.pgerror)

    def _close_db(self):
        if not self._cur.closed:
            self._cur.close()
        if not self._conn.closed:
            self._conn.close()

    def set_meta_data(self, **kwargs):
        """Set meta data of tables if it is different from the default."""
        for k, v in kwargs.items():
            if k not in self._meta_data.keys():
                raise ValueError("set_meta_data: invalid key {}".format(k))
            if not isinstance(v, (str, bool, int)):
                raise ValueError("set_meta_data: invalid value {}".format(v))
            if isinstance(v, str):
                self._meta_data.update({k: psycopg2.sql.Identifier(v)})
            else:
                self._meta_data.update({k: psycopg2.sql.Literal(v)})
        return self._meta_data

    def find_nearest_vertices(self, nodes: List[PgrNode], bbox=None) -> List[PgrNode]:
        """Queries database to find the vertices nearest to the coordinates provided as inputs.

        :param nodes: list of coordinates passed as PgrNode with no id
        :type nodes: list[PgrNode]
        :param bbox:
        :type bbox:
        :return: list of nearest vertices found in the database
        :rtype: list[PgrNode]
        """

        if bbox is None:
            query = """
                SELECT {v_id}, {v_osm_id}, {v_lon}::double precision, {v_lat}::double precision
                FROM {v_table}
                ORDER BY {v_geom} <-> ST_SetSRID(ST_Point(%(longitude)s,%(latitude)s),{srid})
                LIMIT 1
                """
        else:
            query = """
                SELECT {v_id}, {v_osm_id}, {v_lon}::double precision, {v_lat}::double precision
                FROM {v_table}
                WHERE {v_geom} && ST_Expand(
                    (SELECT ST_Collect({v_geom})
                     FROM {v_table}
                     WHERE {v_id} IN %(bbox)s
                    ), 0.001)
                ORDER BY {v_geometry} <-> ST_SetSRID(ST_Point(%(longitude)s,%(latitude)s),{srid})
                LIMIT 1
                """
        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_geom=self._meta_data['vertices_geometry'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude']
        )

        output = []
        for node in nodes:
            try:
                if bbox is None:
                    self._cur.execute(query, {'longitude': node.lon, 'latitude': node.lat})
                else:
                    self._cur.execute(query, {'longitude': node.lon, 'latitude': node.lat, 'bbox': tuple(bbox)})
                results = self._cur.fetchall()
                if len(results) > 0:
                    output.append(
                        PgrNode(id=results[0][0], osm_id=results[0][1], lon=results[0][2], lat=results[0][3])
                    )
                else:
                    print('cannot find nearest vid for ({}, {})'.format(node.lon, node.lat))
                    output.append(None)
            except psycopg2.Error as e:
                print(e.pgerror)
                return []
        return output

    def find_nearest_edge(self, node: PgrNode, bbox=None) -> dict:
        """ finds nearest edge to the passed nodes"""

        if bbox is None:
            query = """
                SELECT {e_source}, {e_source_osm}, {e_x1}, {e_y1}, {e_target}, {e_target_osm}, {e_x2}, {e_y2}, {e_cost},
                 {e_reverse_cost}, {e_length}, {e_way}
                FROM {e_table}
                ORDER BY {e_geom} <-> ST_SetSRID(ST_Point(%(longitude)s,%(latitude)s),{srid})
                LIMIT 1;
                """
        else:
            query = """
                SELECT {e_source}, {e_source_osm}, {e_x1}, {e_y1}, {e_target}, {e_target_osm}, {e_x2}, {e_y2}, {e_cost},
                 {e_reverse_cost}, {e_length}, {e_way}
                FROM {e_table}
                WHERE {e_geom} && ST_Expand(
                    (SELECT ST_Collect({v_geom})
                     FROM {v_table}
                     WHERE {v_id} IN %(bbox)s
                    ), 0.001)
                ORDER BY {e_geom} <-> ST_SetSRID(ST_Point(%(longitude)s,%(latitude)s),{srid})
                LIMIT 1;
                """
        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_geom=self._meta_data['vertices_geometry'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            e_table=self._meta_data['edges_table'],
            e_geom=self._meta_data['edges_geometry'],
            e_id=self._meta_data['edges_id'],
            e_source=self._meta_data['edges_source'],
            e_source_osm=self._meta_data['edges_source_osm'],
            e_x1=self._meta_data['edges_x1'],
            e_y1=self._meta_data['edges_y1'],
            e_target=self._meta_data['edges_target'],
            e_target_osm=self._meta_data['edges_target_osm'],
            e_x2=self._meta_data['edges_x2'],
            e_y2=self._meta_data['edges_y2'],
            e_cost=self._meta_data['edges_cost'],
            e_reverse_cost=self._meta_data['edges_reverse_cost'],
            e_length=self._meta_data['edges_length'],
            e_way=self._meta_data['edges_way_id']
        )
        try:
            if bbox is None:
                self._cur.execute(query, {'longitude': node.lon, 'latitude': node.lat})
            else:
                self._cur.execute(query, {'longitude': node.lon, 'latitude': node.lat, 'bbox': tuple(bbox)})
            results = self._cur.fetchall()
            if results:
                return {'source': PgrNode(id=results[0][0], osm_id=results[0][1], lon=results[0][2], lat=results[0][3]),
                        'target': PgrNode(id=results[0][4], osm_id=results[0][5], lon=results[0][6], lat=results[0][7]),
                        'cost': results[0][8],
                        'reverse_cost': results[0][9],
                        'length': results[0][10],
                        'way': results[0][11]}
            else:
                return {'source': None, 'target': None, 'cost': None, 'reverse_cost': None, 'length': None, 'way': None}

        except psycopg2.Error as e:
            print(e.pgerror)
            return {'source': None, 'target': None, 'cost': None, 'reverse_cost': None, 'length': None, 'way': None}

    def is_vertex(self, node: int, bbox: list = None) -> bool:
        """Checks whether a vertex id is present in the database.

        :param node: vertex id
        :type node: int
        :param bbox:
        :type bbox:
        :return: True if a vertex exists, False otherwise.
        :rtype: bool
        """
        if bbox is None:
            query = """
                SELECT {v_id}::BIGINT
                FROM {v_table}
                WHERE {v_osm_id}=%(node)s
                """
        else:
            query = """
                SELECT {v_id}::BIGINT
                FROM {v_table}
                WHERE {v_osm_id}={node}
                WHERE {v_geom} && ST_Expand(
                    (SELECT ST_Collect({v_geom})
                     FROM {v_table}
                     WHERE {v_id} IN %(bbox)s
                    ), 0.001)
                """
        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_geom=self._meta_data['vertices_geometry'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id']
        )

        try:
            if bbox is None:
                self._cur.execute(query, {'node': node})
            else:
                self._cur.execute(query, {'node': node, 'bbox': bbox})
            results = self._cur.fetchall()
            if results:
                return True
            else:
                return False

        except psycopg2.Error as e:
            print(e.pgerror)
            return False

    def get_vertex(self, node: int) -> PgrNode:
        """Get PgrNode (id and coordinates) given the vertex id.

        :param node: vertex id.
        :type node: int
        :return: PgrNode representation of the vertex including id and coordinates.
        :rtype: PgrNode
        """

        query = """
            SELECT {v_id}, {v_osm_id}, {v_lon}::double precision, {v_lat}:: double precision 
            FROM {v_table}
            WHERE {v_osm_id}=%(vertex_id)s
            """

        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude']
        )

        try:
            self._cur.execute(query, {'vertex_id': node})
            results = self._cur.fetchall()
            if results:
                return PgrNode(id=results[0][0], osm_id=results[0][1], lon=results[0][2], lat=results[0][3])
            else:
                return PgrNode(id=None, osm_id=node, lon=None, lat=None)

        except psycopg2.Error as e:
            print(e.pgerror)
            return PgrNode(id=None, osm_id=node, lon=None, lat=None)

    def is_edge(self, source: int, target: int, bbox) -> bool:
        """Checks whether an edge exists between two nodes.

        :param source: id of the source vertex.
        :type source: int
        :param target: id of the target vertex.
        :type target: int
        :return: True if an edge exists, False otherwise.
        :rtype: bool
        """
        if bbox is None:
            query = """
                SELECT {e_cost}
                FROM {e_table}
                WHERE {e_source}=%(source)s
                AND {e_target}=%(target)s
            """
        else:
            query = """
                SELECT {e_cost}
                FROM {e_table}
                WHERE {e_source}=%(source)s
                AND {e_target}=%(target)s
                WHERE {v_geom} && ST_Expand(
                    (SELECT ST_Collect({v_geom})
                     FROM {v_table}
                     WHERE {v_id} IN %(bbox)s
                    ), 0.001)
                """
        query = psycopg2.sql.SQL(query).format(
            e_table=self._meta_data['edge_table'],
            e_source=self._meta_data['edge_source'],
            e_target=self._meta_data['edge_target'],
            e_cost=self._meta_data['edge_cost']
        )
        # fixme handle the case of bidirectional edges with reverse_cost

        try:
            self._cur.execute(query, {'source': source, 'target': target})
            results = self._cur.fetchall()
            if results:
                return True
            else:
                return False

        except psycopg2.Error as e:
            print(e.pgerror)
            return False

    def node_distance(self, node1: PgrNode, node2: PgrNode) -> float:
        """Get distance between two nodes, in meters.

        :param node1: first node as PgrNode with valid lat and lon
        :type node1: PgrNode
        :param node2: first node as PgrNode with valid lat and lon
        :type node2: PgrNode
        :return: distance in meters
        :rtype: float
        :ref: `https://postgis.net/docs/ST_Distance.html`
        """
        # fixme
        if self._meta_data.get('geometry').strip().lower() == 'the_geom':
            sql = """
            SELECT ST_Distance(
              ST_Transform('SRID={srid};POINT({lon1} {lat1})'::geometry, 3857),
              ST_Transform('SRID={srid};POINT({lon2} {lat2})'::geometry, 3857)
            ) * cosd(42.3521);
            """.format(srid=self._meta_data['srid'], lon1=node1.lon, lat1=node1.lat, lon2=node2.lon, lat2=node2.lat)
        else:  # geography
            sql = """
            SELECT ST_Distance(
                'SRID={srid};POINT({lon1} {lat1})'::geography,
                'SRID={srid};POINT({lon1} {lat1})'::geography
            );
            """.format(srid=self._meta_data['srid'], lon1=node1.lon, lat1=node1.lat, lon2=node2.lon, lat2=node2.lat)

        try:
            self._cur.execute(sql)
            results = self._cur.fetchall()
            return results[0][0]
        except psycopg2.Error as e:
            print(e.pgerror)
            return None

    def dijkstra(self, start_vids: List[int], end_vids: List[int], bbox_vids: List[int] = None) -> pd.DataFrame:
        """Get all-pairs shortest paths with costs among way nodes using pgr_dijkstra function.

        :param start_vids: ids of the start vertices
        :type start_vids: list[int]
        :param end_vids: ids of the end vertices
        :type end_vids: list[int]
        :param bbox_vids:
        :type bbox_vids:
        :return:
        :rtype: pd.DataFrame
        """

        if bbox_vids is None:
            query = """
                SELECT r.*, v.{v_lon}::double precision, v.{v_lat}::double precision
                FROM pgr_dijkstra(
                        'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost
                         FROM {e_table}',
                        (%(sources)s),
                        (%(targets)s),
                        {directed}) as r,
                    {v_table} as v
                WHERE r.node=v.{v_id}
                ORDER BY r.seq;
            """
        else:
            query = """
                SELECT r.*, v.{v_lon}::double precision, v.{v_lat}::double precision
                FROM pgr_dijkstra(
                        'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost
                         FROM {e_table}',
                        (%(sources)s),
                        (%(targets)s),
                        {directed}) as r,
                    {v_table} as v
                WHERE r.node=v.{v_id}
                ORDER BY r.seq;
            """

        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_id=self._meta_data['vertices_id'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude'],
            e_table=self._meta_data['edges_table'],
            e_id=self._meta_data['edges_id'],
            e_source=self._meta_data['edges_source'],
            e_target=self._meta_data['edges_target'],
            e_cost=self._meta_data['edges_cost'],
            e_reverse_cost=self._meta_data['edges_reverse_cost'],
            directed=self._meta_data['directed']
        )

        try:
            self._cur.execute(query, {'sources': (start_vids,), 'targets': (end_vids,)})
            results = self._cur.fetchall()
            output = pd.DataFrame(results, columns=['seq', 'path_seq', 'start_vid', 'end_vid', 'node', 'edge', 'cost',
                                                    'agg_cost', 'v_lon', 'v_lat'])
            return output

        except psycopg2.Error as e:
            print(e.pgerror)
            return pd.DataFrame(columns=['seq', 'path_seq', 'start_vid', 'end_vid', 'node', 'edge', 'cost', 'agg_cost',
                                         'v_lon', 'v_lat'])

    def dijkstra_via(self, start_vid: int, end_vid: int, via_vids: List[int], bbox_vids: List[int] = None) \
            -> pd.DataFrame:
        """Get one-to-one shortest path between start and end through via points, using pgr_dijkstraVia function.

        :param start_vid: id of the start vertex
        :type start_vid: int
        :param end_vid: id of the end vertex
        :type end_vid: int
        :param via_vids: ids of via vertices
        :type via_vids: list[int]
        :param bbox_vids:
        :type bbox_vids:
        :return:
        :rtype: pd.DataFrame
        """
        path_vids = [start_vid] + via_vids + [end_vid]

        if bbox_vids is None:
            query = """
                SELECT r.*, v.{v_osm_id}, v.{v_lon}::double precision, v.{v_lat}::double precision, e.{e_osm_id}, 
                e.{e_length}, e.{e_geom}
                FROM pgr_dijkstraVia(
                        'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost
                         FROM {e_table}',
                        (%(path_vids)s),
                        directed:=false,
                        strict:=true) as r,
                    {v_table} as v,
                    {e_table} as e
                WHERE r.edge=e.{e_id} AND r.node=v.{v_id}
                ORDER BY r.seq;
                """
        else:
            query = """
                SELECT r.*, v.{v_osm_id}, v.{v_lon}::double precision, v.{v_lat}::double precision, e.{e_osm_id}, 
                e.{e_length}, e.{e_geom}
                FROM pgr_dijkstraVia(
                        'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost
                         FROM {e_table}
                         WHERE {e_geom} && ST_Expand(
                            (SELECT ST_Collect({v_geom})
                             FROM {v_table}
                             WHERE {v_id} IN %(bbox_vids)s
                            ), 0.001)',
                        (%(path_vids)s),
                        directed:={directed},
                        strict:=false) as r,
                    {v_table} as v,
                    {e_table} as e
                WHERE r.edge=e.{e_id} AND r.node=v.{v_id}
                ORDER BY r.seq;
            """

        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            v_geom=self._meta_data['vertices_geometry'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude'],
            e_table=self._meta_data['edges_table'],
            e_id=self._meta_data['edges_id'],
            e_osm_id=self._meta_data['edges_way_id'],
            e_source=self._meta_data['edges_source'],
            e_target=self._meta_data['edges_target'],
            e_cost=self._meta_data['edges_cost'],
            e_reverse_cost=self._meta_data['edges_reverse_cost'],
            e_length=self._meta_data['edges_length'],
            e_geom=self._meta_data['edges_geometry'],
            directed=self._meta_data['directed']
        )

        try:
            if bbox_vids is None:
                self._cur.execute(query, {'path_vids': (path_vids,)})
            else:
                self._cur.execute(query, {'path_vids': (path_vids,), 'bbox_vids': tuple(bbox_vids)})
            results = self._cur.fetchall()
            output = pd.DataFrame(results, columns=['seq', 'path_id', 'path_seq', 'start_vid', 'end_vid', 'node',
                                                    'edge', 'cost', 'agg_cost', 'route_agg_cost', 'v_osm_id', 'v_lon',
                                                    'v_lat', 'way_id', 'length', 'geom'])
            return output

        except psycopg2.Error as e:
            print(e.pgerror)
            return pd.DataFrame(columns=['seq', 'path_id', 'path_seq', 'start_vid', 'end_vid', 'node', 'edge', 'cost',
                                         'agg_cost', 'route_agg_cost', 'v_osm_id', 'v_lon', 'v_lat', 'way_id',
                                         'length', 'geom'])

    def astar(self, start_vid: int, end_vid: int, bbox_vids: List[int] = None) -> pd.DataFrame:
        """Get one-to-one shortest path between way nodes using pgr_AStar function.

        :param start_vid: id of the start vertex
        :type start_vid: int
        :param end_vid: id of the end vertex
        :type end_vid: int
        :param bbox_vids:
        :type bbox_vids:
        :return:
        :rtype: pd.DataFrame
        """

        if bbox_vids is None:
            query = """
                SELECT r.*, v.{v_osm_id}, v.{v_lon}::double precision, v.{v_lat}::double precision, e.{e_osm_id}, 
                e.{e_length}
                FROM pgr_AStar(
                         'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost,
                                {e_x1} as x1,
                                {e_y1} as y1,
                                {e_x2} as x2,
                                {e_y2} as y2
                         FROM {e_table}',
                        %(start_vid)s,
                        %(end_vid)s,
                        directed:={directed}
                        ) as r,
                    {v_table} as v,
                    {e_table} as e
                WHERE r.edge=e.{e_id} AND r.node=v.{v_id}
                ORDER BY r.seq;
            """
        else:
            query = """
                SELECT r.*, v.{v_osm_id}, v.{v_lon}::double precision, v.{v_lat}::double precision, e.{e_osm_id}, 
                e.{e_length}
                FROM pgr_AStar(
                         'SELECT {e_id}::BIGINT as id,
                                {e_source}::BIGINT as source,
                                {e_target}::BIGINT as target,
                                {e_cost} as cost,
                                {e_reverse_cost} as reverse_cost,
                                {e_x1} as x1,
                                {e_y1} as y1,
                                {e_x2} as x2,
                                {e_y2} as y2
                         FROM {e_table}
                         WHERE {e_geom} && ST_Expand(
                            (SELECT ST_Collect({v_geom})
                             FROM {v_table}
                             WHERE {v_id} IN %(bbox_vids)s
                            ), 0.001)',
                        %(start_vid)s,
                        %(end_vid)s,
                        directed:={directed}
                        ) as r,
                    {v_table} as v,
                    {e_table} as e
                WHERE r.edge=e.{e_id} AND r.node=v.{v_id}
                ORDER BY r.seq;
            """

        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude'],
            v_geom=self._meta_data['vertices_geometry'],
            e_table=self._meta_data['edges_table'],
            e_id=self._meta_data['edges_id'],
            e_osm_id=self._meta_data['edges_way_id'],
            e_length=self._meta_data['edges_length'],
            e_source=self._meta_data['edges_source'],
            e_target=self._meta_data['edges_target'],
            e_x1=self._meta_data['edges_x1'],
            e_x2=self._meta_data['edges_x2'],
            e_y1=self._meta_data['edges_y1'],
            e_y2=self._meta_data['edges_y2'],
            e_cost=self._meta_data['edges_cost'],
            e_reverse_cost=self._meta_data['edges_reverse_cost'],
            e_geom=self._meta_data['edges_geometry'],
            directed=self._meta_data['directed']
        )

        try:
            if bbox_vids is None:
                self._cur.execute(query, {'start_vid': start_vid, 'end_vid': end_vid})
            else:
                self._cur.execute(query, {'start_vid': start_vid, 'end_vid': end_vid, 'bbox_vids': tuple(bbox_vids)})
            results = self._cur.fetchall()
            output = pd.DataFrame(results, columns=['seq', 'path_seq', 'node', 'edge', 'cost', 'agg_cost', 'v_osm_id',
                                                    'v_lon', 'v_lat', 'e_osm_id', 'e_length'])
            return output

        except psycopg2.Error as e:
            print(e.pgerror)
            return pd.DataFrame(columns=['seq', 'path_seq', 'node', 'edge', 'cost', 'agg_cost', 'v_osm_id', 'v_lon',
                                         'v_lat', 'e_osm_id', 'e_length'])

    def get_route(self, start_node: PgrNode, end_node: PgrNode, via_nodes: List[PgrNode] = None,
                  bbox_nodes: List[PgrNode] = None) -> pd.DataFrame:
        """Get shortest paths from nodes to nodes.

        :param start_node: start location.
        :type start_node: PgrNode
        :param end_node: end location.
        :type end_node: PgrNode
        :param via_nodes: via locations.
        :type via_nodes: List[PgrNode]
        :param bbox_nodes: via locations.
        :type bbox_nodes: List[PgrNode]
        :return:
        :rtype: pd.DataFrame
        """

        if start_node == end_node:
            return pd.DataFrame()

        if bbox_nodes is not None:
            bbox_vertices = self.find_nearest_vertices(bbox_nodes)
            bbox_vids = [bbox_vertices[0].id, bbox_vertices[1].id, bbox_vertices[2].id, bbox_vertices[3].id]
        else:
            bbox_vids = None

        # if start_node.osm_id is None:
        # start_vertex = self.find_nearest_vertices([start_node])[0]
        start_edge = self.find_nearest_edge(start_node)  #, bbox_vids)
        start_source = start_edge['source']
        start_target = start_edge['target']
        # start_cost = start_edge['cost']
        # start_rcost = start_edge['reverse_cost']
        start_length = start_edge['length'] * 1.0e3
        start_way = start_edge['way']
        # else:
        #     start_source = self.get_vertex(start_node.osm_id)
        #     start_cost = 0.0

        # if end_node.osm_id is None:
        # end_vertex = self.find_nearest_vertices([end_node])[0]
        end_edge = self.find_nearest_edge(end_node)  #, bbox_vids)
        end_source = end_edge['source']
        end_target = end_edge['target']
        # end_cost = end_edge['cost']
        # end_rcost = end_edge['reverse_cost']
        end_length = end_edge['length'] * 1.0e3
        end_way = end_edge['way']
        # else:
        #     end_source = self.get_vertex(end_node.osm_id)
        #     end_cost = 0.0

        # routing between vertices
        if via_nodes is None:
            route = self.astar(start_target.id, end_source.id, bbox_vids)
        else:
            via_nodes = [self.get_vertex(n.osm_id) if self.is_vertex(n.osm_id) else n for n in via_nodes]
            via_vids = [n.id for n in via_nodes if n.id is not None]

            if via_vids[0] == start_target.id:
                via_vids.pop(0)
            if via_vids[-1] == end_source.id:
                via_vids.pop(-1)

            main_routing = self.dijkstra_via(start_target.id, end_source.id, via_vids, bbox_vids)
            route = main_routing[['v_osm_id', 'length', 'way_id', 'v_lon', 'v_lat']].rename(
                columns={'v_osm_id': 'source', 'way_id': 'way', 'v_lon': 'source_lon', 'v_lat': 'source_lat'})
            route.length = route.length * 1.0e3
            # note that normally an edge is traversed completely, from start_vid to end_vid, and node = start_vid
            # in some cases (edges with multiple inputs/outputs?) there are multiple steps with the same start_vid and
            # end_vid, and the node matched start_vid in the first step (or end_vid in the last step). This is signalled
            # by path_seq becoming greater than 1
            # v_som_id is just the osm id of node, and in the returned route we output as edges any source/target pair
            # in the node path of v_osm_id; in most cases these are actual graph edges, but if the situation above
            # happens they are segments of an edge geometry
            route['target'] = route.source.shift(-1)
            route['target_lon'] = route.source_lon.shift(-1)
            route['target_lat'] = route.source_lat.shift(-1)
            route.target.iloc[-1] = end_source.osm_id
            route.target_lon.iloc[-1] = end_source.lon
            route.target_lat.iloc[-1] = end_source.lat

            # fixme this is really needed only when there are no via_nodes - otherwise I could get the direction from
            #  the first and last via_node
            # fix start edge and cost if needed (assumes start_target was set as route source)
            if not (route.target.iloc[0] == start_source.osm_id):
                # need to add as first edge (start_source -> start_target)
                # add length to cost, should add only (length - abscissa)
                route = pd.concat([pd.DataFrame([{'source': start_source.osm_id, 'way': start_way,
                                                  'length': start_length, 'source_lon': start_source.lon,
                                                  'source_lat': start_source.lat, 'target': start_target.osm_id,
                                                  'target_lon': start_target.lon, 'target_lat': start_target.lat}]),
                                   route]).reset_index()
            # else:
            # first edge is already (start_target -> start_source)
            # path stays same
            # cost stays same, should subtract (length - abscissa)

            # fix end edge and cost if needed (assumes end_source was set as route target)
            if not (route.source.iloc[-1] == end_target.osm_id):
                # need to add as last edge (end_source -> end_target)
                # add length to cost, should add abscissa
                route = pd.concat([route,
                                   pd.DataFrame([{'source': end_source.osm_id, 'way': end_way, 'length': end_length,
                                                 'source_lon': end_source.lon, 'source_lat': end_source.lat,
                                                 'target': end_target.osm_id, 'target_lon': end_target.lon,
                                                 'target_lat': end_target.lat}])]).reset_index()
            # else:
            # last edge is already end_target -> end_source
            # path stays same
            # cost stays same, should subtract abscissa

            # fix node/way ids type
            route.source = route.source.astype('int')
            route.target = route.target.astype('int')
            route.way = route.way.astype('int')

            # map the pgrouting route edges to the provided via nodes. Every row in via_map is an edge between two
            # via_nodes (source and target) that correspond to a pgrouting vertex. If there are any via_nodes in between
            # these two nodes, they are listed in the nodes column. The nodes columns is then added to the output route.
            # When the path between two via_nodes that are pgrouting vertices includes more than one edge, the nodes
            # column in the corresponding rows is left empty (None)
            # fixme add start/end to via_nodes
            via_map = [{'source': via_nodes[0].osm_id, 'target': None, 'nodes': []}]
            for i, n in enumerate(via_nodes[1:]):
                if i == len(via_nodes) - 2:
                    via_map[-1]['target'] = n.osm_id
                elif n.id is not None:
                    via_map[-1]['target'] = n.osm_id
                    via_map.append({'source': n.osm_id, 'target': None, 'nodes': []})
                else:
                    via_map[-1]['nodes'].append(n.osm_id)
            route = route.merge(pd.DataFrame(via_map).drop_duplicates(subset=['source', 'target']),
                                how='left', left_on=['source', 'target'], right_on=['source', 'target'])

        return route

    def driving_distance(self, start_vid, distance_km):
        # pgr_drivingDistance(edges_sql, start_vid, distance [, directed])

        distance_deg = distance_km / 111.0
        query = """
            SELECT r.*, v.{v_osm_id}, v.{v_lon}::double precision, v.{v_lat}::double precision
            FROM pgr_drivingDistance(
                    'SELECT {e_id}::BIGINT as id,
                            {e_source}::BIGINT as source,
                            {e_target}::BIGINT as target,
                            {e_cost} as cost,
                            {e_reverse_cost} as reverse_cost
                     FROM {e_table}',
                    %(start_vid)s::BIGINT,
                    %(distance)s::FLOAT) as r,
                    {v_table} as v
            WHERE r.node=v.{v_id}
            ORDER BY r.seq;
            """
        # SELECT id, source, target, cost, reverse_cost FROM edge_table
        query = psycopg2.sql.SQL(query).format(
            srid=self._meta_data['srid'],
            v_table=self._meta_data['vertices_table'],
            v_id=self._meta_data['vertices_id'],
            v_osm_id=self._meta_data['vertices_osm_id'],
            v_lon=self._meta_data['vertices_longitude'],
            v_lat=self._meta_data['vertices_latitude'],
            v_geom=self._meta_data['vertices_geometry'],
            e_table=self._meta_data['edges_table'],
            e_id=self._meta_data['edges_id'],
            e_osm_id=self._meta_data['edges_way_id'],
            e_length=self._meta_data['edges_length'],
            e_source=self._meta_data['edges_source'],
            e_target=self._meta_data['edges_target'],
            e_x1=self._meta_data['edges_x1'],
            e_x2=self._meta_data['edges_x2'],
            e_y1=self._meta_data['edges_y1'],
            e_y2=self._meta_data['edges_y2'],
            e_cost=self._meta_data['edges_cost'],
            e_reverse_cost=self._meta_data['edges_reverse_cost'],
            e_geom=self._meta_data['edges_geometry'],
            directed=self._meta_data['directed']
        )

        try:
            self._cur.execute(query, {'start_vid': start_vid, 'distance': distance_deg})
            results = self._cur.fetchall()
            output = pd.DataFrame(results, columns=['seq', 'node', 'edge', 'cost', 'agg_cost', 'v_osm_id',
                                                    'v_lon', 'v_lat'])
            return output

        except psycopg2.Error as e:
            print(e.pgerror)
            return pd.DataFrame(columns=['seq', 'node', 'edge', 'cost', 'agg_cost', 'v_osm_id', 'v_lon',
                                         'v_lat'])

    def edges_to_csv(self, file_path: str, columns: List[str] = None):
        query = """COPY (SELECT {e_columns} FROM {e_table}) TO STDOUT WITH CSV HEADER""".format(
            e_table=self._meta_data['edges_table'].string,
            e_columns=','.join(columns)
        )
        with open(file_path, 'w') as f:
            self._cur.copy_expert(query, f)

    def edges_to_csv2(self, file_path: str, columns: List[str] = None):
        query = """COPY (SELECT {e_columns} FROM {e_table}) TO STDOUT WITH CSV HEADER""".format(
            e_table=self._meta_data['edges_table'].string,
            e_columns=','.join(columns)
        )
        with open(file_path, 'w') as f:
            self._cur.copy_expert(query, f)
