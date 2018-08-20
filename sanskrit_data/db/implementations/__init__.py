# Sphinx uses the all list to check every module is loaded, but in some cases it is not and a warning is generated:
#   missing attribute mentioned in :members: or __all__: module
"""Allows users to do ``from xyz import *`` """
__all__ = ["couchdb", "mongodb"]
