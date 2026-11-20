import benodb as bdb
import os
import pytest
from benodb import PyNessieCatalog

def test_load_default_catalog():
    print("Testing load_default_catalog...")
    
    # create a dummy benodb.toml in current directory
    config_content = """
    catalog_type = "nessie"
    [config]
    url = "http://localhost:19120"
    """
    
    with open("benodb.toml", "w") as f:
        f.write(config_content)
        
    try:
        catalog = bdb.load_default_catalog()
        assert isinstance(catalog, PyNessieCatalog)
        print("Default catalog loading passed.")
    finally:
        if os.path.exists("benodb.toml"):
            os.remove("benodb.toml")

if __name__ == "__main__":
    test_load_default_catalog()
