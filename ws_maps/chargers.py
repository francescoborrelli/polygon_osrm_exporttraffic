#  Copyright (c) 2020. AV Connect Inc.
"""This module provides access to charging stations data. """

import requests
from functools import reduce

BASE_URL = 'https://developer.nrel.gov/api/alt-fuel-stations/v1/nearby-route.json'
DISTANCE = .5
MAX_RETURNED = 30
NETWORKS = "ChargePoint Network, eCharge Network, Electrify America, Blink Network, OpConnect, EV Connect, eVgo Network, Greenlots, Non-Networked, Volta"


def get_multipoints(waypoints):
    f = lambda p, c: "{} {} {},".format(p, c[1], c[0])
    list = reduce(f, waypoints, "")[:-1]

    return "LINESTRING({})".format(list)


def get_charge_stations(config, waypoints):
    """
    param: route - array of  lat,lon pairs [[lat1,lon1],[lat2,lon2] etc
    """
    waypoint_string = get_multipoints(waypoints)
    params = {
        'api_key': config['NREL_KEY'],
        'distance': DISTANCE,
        'status': 'E',
        'access': 'public',
        'fuel_type': 'ELEC',
        'limit': MAX_RETURNED,
        'ev_network': NETWORKS,
        'format':'JSON'
    }
    data = {
        'route': waypoint_string
    }
    response = requests.post(BASE_URL, params=params, data=data)
    return response.json()

