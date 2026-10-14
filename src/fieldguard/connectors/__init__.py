"""Remote data connectors (KoboToolbox API v2, ODK Central OData, SurveyCTO server API).

Credentials come only from environment variables. Each connector returns the downloaded records; use
:func:`fieldguard.connectors.base.write_raw_csv` to save them in the export layout that ``fieldguard run`` reads.
"""

from fieldguard.connectors.base import ConnectorError, MissingCredentialError, PageCache, write_raw_csv
from fieldguard.connectors.kobo import fetch_kobo
from fieldguard.connectors.kobo import settings_from_env as kobo_settings_from_env
from fieldguard.connectors.odk import fetch_odk
from fieldguard.connectors.odk import settings_from_env as odk_settings_from_env
from fieldguard.connectors.surveycto import fetch_surveycto
from fieldguard.connectors.surveycto import settings_from_env as surveycto_settings_from_env

__all__ = [
    "ConnectorError",
    "MissingCredentialError",
    "PageCache",
    "fetch_kobo",
    "fetch_odk",
    "fetch_surveycto",
    "kobo_settings_from_env",
    "odk_settings_from_env",
    "surveycto_settings_from_env",
    "write_raw_csv",
]
