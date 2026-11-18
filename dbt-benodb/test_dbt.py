import sys
try:
    from dbt.adapters.benodb.connections import BenoDBConnectionManager, BenoDBCredentials
    import benodb

    session = benodb.Session()
    result = session.sql("SELECT 1 AS id")
    print(result.to_pandas())
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
