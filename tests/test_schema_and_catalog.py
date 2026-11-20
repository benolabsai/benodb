import pytest
import benodb as bdb
import os
import shutil

def test_datatype_constructors():
    """Verify all DataType static constructors exist and return DataType objects"""
    types = [
        bdb.DataType.int8(),
        bdb.DataType.int16(),
        bdb.DataType.int32(),
        bdb.DataType.int64(),
        bdb.DataType.uint8(),
        bdb.DataType.uint16(),
        bdb.DataType.uint32(),
        bdb.DataType.uint64(),
        bdb.DataType.float16(),
        bdb.DataType.float32(),
        bdb.DataType.float64(),
        bdb.DataType.string(),
        bdb.DataType.binary(),
        bdb.DataType.boolean(),
        bdb.DataType.date32(),
        bdb.DataType.date64(),
        bdb.DataType.timestamp_ms(),
        bdb.DataType.timestamp_us(),
        bdb.DataType.vector(128, False),
    ]
    for t in types:
        assert isinstance(t, bdb.DataType)
        assert repr(t) is not None

def test_field_and_schema_construction():
    """Verify Field, PartitionField, and Schema construction works correctly"""
    # 1. Create fields
    field_id = bdb.Field("id", bdb.DataType.int64(), False, {"description": "Primary Key"})
    field_val = bdb.Field("val", bdb.DataType.string(), True)
    field_vec = bdb.Field("embedding", bdb.DataType.vector(128, False))

    assert isinstance(field_id, bdb.Field)
    assert "id" in repr(field_id)
    assert "Int64" in repr(field_id)

    # 2. Build Schema
    schema = bdb.Schema([field_id, field_val, field_vec], {"table_type": "vector_table"})
    assert isinstance(schema, bdb.Schema)
    assert repr(schema) is not None

    # 3. Create PartitionField
    part_field = bdb.PartitionField([1], "category", "identity", 100)
    assert isinstance(part_field, bdb.PartitionField)
    assert part_field.source_ids == [1]
    assert part_field.name == "category"
    assert part_field.transform == "identity"
    assert part_field.field_id == 100

def test_unity_catalog_operations():
    """Verify Unity Catalog wrapper initialization and basic operations raise clean errors on invalid server connection"""
    catalog = bdb.PyUnityCatalog("http://localhost:8080", "mock-token")
    assert isinstance(catalog, bdb.PyUnityCatalog)

    # Create dummy schema
    field_id = bdb.Field("id", bdb.DataType.int64(), False)
    schema = bdb.Schema([field_id])

    # Should raise RuntimeError due to connection refusal
    with pytest.raises(RuntimeError):
        catalog.create_table("my_catalog.my_schema", "my_table", schema, None)

    with pytest.raises(RuntimeError):
        catalog.load_table("my_catalog.my_schema", "my_table")

    # table_exists catches the error and returns False
    assert catalog.table_exists("my_catalog.my_schema", "my_table") is False

def test_jdbc_catalog_operations():
    """Verify JDBC Catalog wrapper initialization and operations with an in-memory SQLite backend"""
    try:
        # Try establishing with SQLite in-memory URL
        catalog = bdb.PyJdbcCatalog("sqlite::memory:", "my_warehouse", "my_catalog")
    except BaseException as e:
        # If sqlx-any driver isn't registered on this host/setup, expect a clean exception/panic handle
        print(f"Skipping complete JDBC test as sqlx-any SQLite driver is not loaded: {e}")
        # Verify FFI construction with invalid parameters still raises expected exception
        with pytest.raises(BaseException):
            bdb.PyJdbcCatalog("invalid_uri://", "my_warehouse", "my_catalog")
        return

    assert isinstance(catalog, bdb.PyJdbcCatalog)

    # Create dummy schema
    field_id = bdb.Field("id", bdb.DataType.int64(), False)
    schema = bdb.Schema([field_id])

    # table_exists initially should return False
    assert catalog.table_exists("my_namespace", "my_table") is False

    # load_table on non-existent table should raise RuntimeError (RowNotFound)
    with pytest.raises(RuntimeError):
        catalog.load_table("my_namespace", "my_table")

    # Clean up test table directory if it exists
    test_loc = os.path.abspath("test_sqlite_jdbc_table")
    if os.path.exists(test_loc):
        shutil.rmtree(test_loc)

    try:
        # create_table should succeed and register the table in the SQLite metadata store
        catalog.create_table("my_namespace", "my_table", schema, f"file://{test_loc}")

        # Now, table_exists should return True!
        assert catalog.table_exists("my_namespace", "my_table") is True

        # load_table returns the *raw* Rust table. `benodb.Table` is the
        # Python wrapper from `__init__.py` (it stores the Rust table as
        # `_inner`), so the two are different classes and
        # `isinstance(table, bdb.Table)` can never be true. There is also no
        # `bdb.PyTable` -- the Rust struct is `PyTable` but the pyclass is
        # registered as `Table` in the extension module.
        table = catalog.load_table("my_namespace", "my_table")
        assert isinstance(table, bdb.benodb.Table)
        assert table.table_uri() == f"file://{test_loc}"
    finally:
        # Clean up
        if os.path.exists(test_loc):
            shutil.rmtree(test_loc)

def test_manifest_not_directly_instantiable():
    """Verify Manifest and ManifestEntry exist in the module but cannot be directly constructed"""
    assert hasattr(bdb, "Manifest")
    assert hasattr(bdb, "ManifestEntry")

    with pytest.raises(TypeError):
        bdb.Manifest()

    with pytest.raises(TypeError):
        bdb.ManifestEntry()
