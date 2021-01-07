import boto3
import json
from ws_maps.config import Config


class Blob:
    """
    Persist and get blob data from s3.
    Can only run if you have s3 credentials in your home directory (.aws)
    """

    def __init__(self, bucket_name=None):
        self._config = Config()
        self._s3 = boto3.resource('s3')
        self._client = boto3.client('s3')
        if not bucket_name:
            self._bucket_name = self._config.s3_bucket
        else:
            self._bucket_name = bucket_name

    def set(self, key, value):
        """

        :param key:
        :param value:
        :return:
        """
        self._s3.Object(self._bucket_name, key).put(Body=json.dumps(value))

    def get(self, key):
        """

        :param key:
        :return:
        """
        print(self._bucket_name)
        json_data = self._s3.Object(self._bucket_name, key).get()["Body"].read()
        return json.loads(json_data)

    def list_keys(self, prefix=None):
        """

        :param prefix:
        :return:
        """
        pg = self._client.get_paginator('list_objects_v2')
        if not prefix:
            pger = pg.paginate(Bucket=self._bucket_name)
        else:
            pger = pg.paginate(Bucket=self._bucket_name, Prefix=prefix)
        for page in pger:
            lines = page['Contents']
            for line in lines:
                yield line['Key']

