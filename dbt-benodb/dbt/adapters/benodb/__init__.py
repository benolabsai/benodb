from dbt.adapters.base import AdapterPlugin
from dbt.include import benodb

from .connections import BenoDBConnectionManager
from .connections import BenoDBCredentials
from .impl import BenoDBAdapter

Plugin = AdapterPlugin(
    adapter=BenoDBAdapter,
    credentials=BenoDBCredentials,
    include_path=benodb.PACKAGE_PATH,
)
