import tempfile
"""
Cross-Engine Compatibility Tests for Iceberg V2/V3 Features

Tests that BenoStreamDB-written tables can be read by:
- Apache Spark
- Trino
- Other Iceberg-compatible engines

Prerequisites:
- Docker (for Spark/Trino containers)
- benodb installed
"""

import benodb as bdb
import pandas as pd
import pytest
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import json

# Test data directory
TEST_DIR = (Path(tempfile.gettempdir()) / "benodb_compat_tests")
TEST_DIR.mkdir(exist_ok=True)

# The per-PR `ecosystem` job cannot read Iceberg-format tables from the pip
# `pyspark` package (it ships no Iceberg JAR), so these tests skip there. The
# `Iceberg Interop (JVM)` workflow sets the variables below to turn the skips
# into hard failures and to supply the Iceberg Spark runtime from Maven:
#
#   BDB_REQUIRE_JVM_INTEROP=1
#   BDB_SPARK_ICEBERG_JAR=<local path to the pinned runtime jar>
#   BDB_SPARK_ICEBERG_PACKAGES=org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.11.0
#
# The workflow downloads and checksum-verifies the jar and passes it with
# `--jars`. `--packages` stays as the fallback for local runs, but it resolves
# through Ivy, and a restored `~/.ivy2/cache` answered a resolution with zero
# repository searches and a spurious "not found" for an artifact Maven Central
# was serving.
#
# The Iceberg runtime artifact is keyed by Spark *minor series*, and Maven
# Central publishes exactly: 3.4, 3.5, 4.0, 4.1 (each `_2.12` and `_2.13`).
# There is no 4.2 runtime, so Spark 4.2.x cannot be paired with a supported
# Iceberg build — pin 3.5.2 (+ `3.5_2.12`) or 4.0/4.1 (+ `_2.13`). Iceberg 1.11
# is used rather than 1.6 because V3 deletion vectors and row lineage need a
# recent runtime.
REQUIRE_JVM_INTEROP = os.environ.get("BDB_REQUIRE_JVM_INTEROP") == "1"
SPARK_ICEBERG_PACKAGES = os.environ.get(
    "BDB_SPARK_ICEBERG_PACKAGES",
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.11.0",
)
SPARK_ICEBERG_JAR = os.environ.get("BDB_SPARK_ICEBERG_JAR", "")


def _spark_runtime_args() -> list[str]:
    """spark-submit arguments that put the Iceberg runtime on the classpath.

    A pre-fetched jar wins because it needs no resolution. Without one, fall
    back to Ivy through `--packages`, which is what local runs use.
    """
    if SPARK_ICEBERG_JAR:
        if not Path(SPARK_ICEBERG_JAR).is_file():
            _require_or_skip(f"BDB_SPARK_ICEBERG_JAR is missing: {SPARK_ICEBERG_JAR}")
        return ["--jars", SPARK_ICEBERG_JAR]
    return ["--packages", SPARK_ICEBERG_PACKAGES]


def _latest_metadata(table_path: str) -> dict | None:
    """Return the parsed newest `metadata/v{N}.metadata.json`, or `None`."""
    metadata_dir = Path(table_path) / "metadata"
    if not metadata_dir.is_dir():
        return None
    best = None
    for p in metadata_dir.iterdir():
        name = p.name
        if name.startswith("v") and name.endswith(".metadata.json"):
            try:
                best = max(best or 0, int(name[1 : -len(".metadata.json")]))
            except ValueError:
                continue
    if best is None:
        return None
    try:
        return json.loads((metadata_dir / f"v{best}.metadata.json").read_text())
    except (ValueError, OSError):
        return None


def _table_format_version(table_path: str) -> int | None:
    """Read `format-version` from the table's latest metadata.json."""
    metadata = _latest_metadata(table_path)
    if metadata is None:
        return None
    try:
        return int(metadata["format-version"])
    except (KeyError, TypeError, ValueError):
        return None


def _spark_submit_available() -> bool:
    return shutil.which("spark-submit") is not None


def _require_or_skip(reason: str) -> None:
    """Skip locally / per-PR, but fail in the interop workflow."""
    if REQUIRE_JVM_INTEROP:
        pytest.fail(f"JVM interop is required but unavailable: {reason}")
    pytest.skip(reason)

class TestSparkCompatibility:
    """Test BenoDB → Spark compatibility"""
    
    def setup_method(self):
        """Create and commit the test table with V2 features.

        The commit is load bearing, for the reason spelled out in the Trino
        class: `write_pandas` buffers into the write ahead log, so an
        uncommitted table holds no Parquet and no `metadata/version-hint.text`.
        Spark then has no Iceberg table to open and reports
        TABLE_OR_VIEW_NOT_FOUND, which names the directory as a namespace and
        sends the reader looking for a catalog problem that does not exist.
        """
        self.table_path = str(TEST_DIR / "spark_test_table")
        # Removing the directory first keeps each run at one committed state;
        # a leftover table would be appended to and the row count would be the
        # sum of every run on the machine.
        shutil.rmtree(self.table_path, ignore_errors=True)
        self.table = bdb.Table(self.table_path)
        
        # Write test data with V2 features
        df = self._fixture_frame()
        
        # Configure V2 features
        self.table.set_sort_order(["timestamp", "id"], ascending=[False, True])
        self.table.write_pandas(df)
        self.table.commit()

    @staticmethod
    def _fixture_frame():
        return pd.DataFrame({
            "id": range(1000),
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1h"),
            "value": range(1000, 2000),
            "category": ["A", "B", "C"] * 333 + ["A"]
        })

    def _write_v3_fixture(self) -> str:
        """Write a format-version 3 table beside the V2 fixture and commit it.

        `format_version` has to be set before the first write: it is the field
        the engine stamps into `metadata/v1.metadata.json`, and once the table
        has committed at 2 there is nothing to promote.
        """
        path = str(TEST_DIR / "spark_test_table_v3")
        shutil.rmtree(path, ignore_errors=True)
        table = bdb.Table(path)
        table.format_version = 3
        table.set_sort_order(["timestamp", "id"], ascending=[False, True])
        table.write_pandas(self._fixture_frame())
        table.commit()
        return path
    
    def test_spark_read_basic(self):
        """Test Spark can read a BenoDB table as an Iceberg table."""
        if not _spark_submit_available():
            _require_or_skip("spark-submit not available")
        # The pip-installed pyspark lacks the Iceberg JAR, so plain CI skips.
        # The interop workflow sets BDB_REQUIRE_JVM_INTEROP=1 and lets
        # spark-submit fetch the runtime from Maven via --packages.
        if not REQUIRE_JVM_INTEROP and (
            os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
        ):
            pytest.skip("pyspark pip package lacks Iceberg JAR in CI")

        import textwrap
        spark_script = textwrap.dedent(f"""\
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \\
            .appName("BenoDB Compat Test") \\
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
            .config("spark.sql.catalog.local.type", "hadoop") \\
            .config("spark.sql.catalog.local.warehouse", "{self.table_path}") \\
            .getOrCreate()

        df = spark.read.format("iceberg").load("{self.table_path}")
        assert df.count() == 1000
        print("✅ Spark read successful")
        """)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(spark_script)
            script_path = f.name
        try:
            cmd = ["spark-submit", "--master", "local[*]"]
            if REQUIRE_JVM_INTEROP:
                cmd += _spark_runtime_args()
            cmd.append(script_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if "SparkClassNotFoundException" in result.stdout or \
                   "DATA_SOURCE_NOT_FOUND" in result.stdout:
                    _require_or_skip("Iceberg Spark runtime JAR not available")
                assert result.returncode == 0, \
                    f"Spark submit failed:\n{result.stderr}\n{result.stdout}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def test_spark_read_v3_metadata(self):
        """Spark must read a V3 table and resolve its row-lineage columns.

        Three facts this pins down, each of which the earlier version of this
        test got wrong or never reached.

        The fixture was never format-version 3, so the old version skipped on
        every run. `_write_v3_fixture` now builds a V3 table next to the V2 one.

        The lineage columns are hidden metadata columns, not schema columns:
        `spark.read.format("iceberg").load(path)` returns only the data columns
        even for a V3 table, so the old `set(df.columns)` assertion could not
        pass against a real table. Resolution has to be checked through SQL.

        Gating is a property of the runtime, not of the spec, so this test does
        not assert it. Against iceberg-spark-runtime 1.11, Spark 4.0 rejects
        `_row_id` on a V2 table with UNRESOLVED_COLUMN while Spark 3.5 resolves
        it on V2 as well. Asserting rejection would therefore pass on one lane
        and fail on the other, so the script records which happened and the
        assertions cover the parts that hold on both: the V3 columns resolve,
        the row count is right, and the engine wrote the row-id cursor into
        table metadata.

        The values come back NULL for this write path. The writer sets the
        snapshot-level `first-row-id`, but Iceberg derives `_row_id` as
        `first_row_id + row_position` from the per-file entry, and there is no
        per-file base here. That is a writer gap, not a reader gap, so this test
        pins resolution and row counts and does not assert non-NULL values.

        Because the catalog resolves identifiers as `<warehouse>/<name>`, the
        warehouse is the fixture directory and the table is addressed as
        `local.spark_test_table_v3`.
        """
        if not REQUIRE_JVM_INTEROP:
            pytest.skip("V3 lineage read needs the Iceberg runtime; run the interop workflow")
        if not _spark_submit_available():
            _require_or_skip("spark-submit not available")

        v3_path = self._write_v3_fixture()
        version = _table_format_version(v3_path)
        assert version == 3, (
            f"`format_version = 3` did not reach the metadata; it says {version}"
        )
        # V3 lineage state lives in table metadata, not in data columns. A
        # `format-version` of 3 with no row-id cursor would mean the engine
        # stamped the version but never tracked identity, which is the failure
        # the Spark-side assertions cannot see.
        metadata = _latest_metadata(v3_path) or {}
        next_row_id = metadata.get("next-row-id")
        assert isinstance(next_row_id, int) and next_row_id >= 1000, (
            f"V3 metadata carries no row-id cursor; next-row-id={next_row_id!r}"
        )
        warehouse = str(TEST_DIR)

        import textwrap
        spark_script = textwrap.dedent(f"""\
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \\
            .appName("BenoDB V3 Lineage Test") \\
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
            .config("spark.sql.catalog.local.type", "hadoop") \\
            .config("spark.sql.catalog.local.warehouse", "{warehouse}") \\
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        rows = spark.table("local.spark_test_table_v3").count()
        cols = spark.sql(
            "SELECT _row_id, _last_updated_sequence_number "
            "FROM local.spark_test_table_v3 LIMIT 1"
        ).columns
        print("V3ROWS", rows)
        print("V3COLS", ",".join(sorted(cols)))

        try:
            spark.sql("SELECT _row_id FROM local.spark_test_table LIMIT 1").collect()
        except Exception:
            print("V2ROWID rejected")
        else:
            print("V2ROWID accepted")
        """)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(spark_script)
            script_path = f.name
        try:
            result = subprocess.run(
                ["spark-submit", "--master", "local[*]", *_spark_runtime_args(), script_path],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Spark V3 lineage read failed:\n{result.stderr}\n{result.stdout}"
            )
            out = result.stdout
            assert "V3ROWS 1000" in out, f"V3 row count wrong; saw\n{out}"
            assert "_row_id" in out and "_last_updated_sequence_number" in out, (
                f"V3 lineage columns did not resolve; saw\n{out}"
            )
            assert "V2ROWID" in out, f"V2 probe did not run; saw\n{out}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)


class TestTrinoCompatibility:
    """Test BenoStreamDB → Trino compatibility"""
    
    def setup_method(self):
        """Create and commit the test table.

        The commit is not optional. `write_pandas` buffers into the write ahead
        log, so a table that is never committed holds no Parquet and no
        `metadata/version-hint.text`. Trino's `register_table` then has nothing
        to read, and the test fails four lines later with TABLE_NOT_FOUND rather
        than with the registration error that caused it.
        """
        self.table_path = str(TEST_DIR / "trino_test_table")
        # A table left behind by an earlier run would be appended to, so the row
        # count assertion would be checking the sum of every run on this machine.
        # Removing the directory makes each run start from one committed state.
        shutil.rmtree(self.table_path, ignore_errors=True)
        self.table = bdb.Table(self.table_path)

        df = pd.DataFrame({
            "user_id": range(500),
            "action": ["click", "view", "purchase"] * 166 + ["click", "view"],
            "amount": [i * 1.5 for i in range(500)]
        })

        self.table.write_pandas(df)
        self.table.commit()
    
    def test_trino_read_basic(self):
        """Trino must read a BenoDB table as an Iceberg table.

        Requires a Trino server whose Iceberg connector is a `hive_metastore`
        catalog over the same directory tree (the interop workflow starts one).
        Skipped everywhere else.

        A metastore is required because `register_table` is rejected for Nessie
        catalogs, and Trino 482 removed the `hadoop` catalog type that once read
        a directory with no metastore at all.
        """
        host = os.environ.get("TRINO_HOST")
        if not host:
            _require_or_skip("TRINO_HOST is not set")
        try:
            import trino  # noqa: F401
        except ImportError:
            _require_or_skip("the `trino` Python client is not installed")

        import trino

        conn = trino.dbapi.connect(
            host=host,
            port=int(os.environ.get("TRINO_PORT", "8080")),
            user="ci",
            catalog="iceberg",
            schema="default",
        )
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS iceberg.default")
        # A bare path fails with "No factory for location: /tmp/.../metadata",
        # because the connector resolves locations through a filesystem factory
        # keyed on a scheme; `file:` plus fs.hadoop.enabled is what the interop
        # catalog sets. The call is deliberately not wrapped in a bare `except`:
        # swallowing it is what turned every registration failure into a
        # TABLE_NOT_FOUND on the next line.
        location = Path(self.table_path).as_uri()
        from trino.exceptions import TrinoUserError

        try:
            cur.execute(
                "CALL iceberg.system.register_table("
                "schema_name => 'default', table_name => 'trino_test_table', "
                f"table_location => '{location}')"
            )
        except TrinoUserError as exc:
            # A rerun on the same machine finds the name already registered at
            # this same location, which setup_method has just rewritten from
            # scratch, so the stale entry is correct. Anything else is a real
            # failure and has to surface: a blanket `except: pass` here is what
            # used to turn registration errors into TABLE_NOT_FOUND.
            #
            # DROP TABLE is not the alternative it looks like. On an Iceberg
            # catalog it deletes the table's data files, so it removes the very
            # fixture that was just written.
            if exc.error_name != "ALREADY_EXISTS":
                raise
        cur.execute('SELECT count(*) FROM iceberg.default."trino_test_table"')
        (count,) = cur.fetchone()
        assert count == 500, f"Trino saw {count} rows, expected 500"


class TestV2Features:
    """Test V2 feature compatibility"""
    
    def test_sort_order_metadata(self):
        """Verify sort order is written to metadata"""
        table_path = str(TEST_DIR / "sort_order_test")
        table = bdb.Table(table_path)
        
        df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        table.set_sort_order(["a"], ascending=[True])
        table.write_pandas(df)
        table.commit()
        
        # Verify sort order in manifest
        manifest_path = Path(table_path) / "_manifest"
        assert manifest_path.exists()

        # Load v1.json to verify sort order metadata
        import json
        v1_path = manifest_path / "v1.json"
        assert v1_path.exists()
        with open(v1_path) as f:
            v1_data = json.load(f)
        
        # Check sort_orders field in v1.json
        assert len(v1_data.get("sort_orders", [])) > 0
        sort_order = v1_data["sort_orders"][0]
        assert len(sort_order.get("fields", [])) > 0
        field = sort_order["fields"][0]
        assert field.get("source-id") == 1
        assert field.get("direction") == "asc"

        # Load metadata/v1.metadata.json
        meta_path = Path(table_path) / "metadata" / "v1.metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta_data = json.load(f)
        assert len(meta_data.get("sort-orders", [])) > 0
        meta_sort_order = meta_data["sort-orders"][0]
        assert len(meta_sort_order.get("fields", [])) > 0
        meta_field = meta_sort_order["fields"][0]
        assert meta_field.get("source-id") == 1
        assert meta_field.get("direction") == "asc"
    
    def test_partition_evolution_metadata(self):
        """Verify partition spec evolution is tracked"""
        import uuid
        table_path = str(TEST_DIR / f"partition_evolution_test_{uuid.uuid4().hex}")
        table = bdb.Table(table_path)
        
        import pyarrow as pa
        import datetime
        dates = pa.array([datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)], type=pa.date32())
        values = pa.array(range(100))
        batch = pa.RecordBatch.from_arrays([dates, values], names=["date", "value"])
        # Set initial partition spec
        table.set_partition_spec([
            {"source_id": 1, "field_id": 1000, "name": "date", "transform": "day"}
        ])
        table.write([batch])
        
        # Verify partition_specs history is tracked
        # (This would check the manifest JSON)
    
    def test_ndv_statistics(self):
        """Verify NDV statistics are computed"""
        table_path = str(TEST_DIR / "ndv_test")
        table = bdb.Table(table_path)
        
        # Create data with known cardinality
        df = pd.DataFrame({
            "id": list(range(100)) * 10,  # 100 distinct values
            "category": ["A", "B", "C"] * 333 + ["A"]  # 3 distinct values
        })
        
        table.write_pandas(df)
        
        # Verify distinct_count is computed
        # (This would check the manifest statistics)


class TestV3Features:
    """Test V3 feature compatibility"""
    
    def test_row_lineage_columns(self):
        """Verify V3 metadata columns are added"""
        table_path = str(TEST_DIR / "v3_lineage_test")
        table = bdb.Table(table_path)
        
        df = pd.DataFrame({"value": [1, 2, 3]})
        table.write_pandas(df)
        
        # Read back and verify metadata columns
        result = table.to_pandas()
        
        # Check for V3 columns (if format_version >= 3)
        # assert "_row_id" in result.columns
        # assert "_last_updated_sequence_number" in result.columns
    
    def test_default_values_schema(self):
        """Verify default values are stored in schema"""
        # This would test that initial_default and write_default
        # are properly serialized in the schema metadata
        pass


class TestSparkOutOfBandAppend:
    """#100: an external writer appends between index builds.

    Spark owns the upstream table and never sees the overlay prefix; BenoDB
    mounts it read only and writes only its sidecars. Two properties are under
    test, and together they are why serving a pinned overlay is safe:

    * before a sync the overlay reports the snapshot it imported and returns
      exactly that snapshot's rows, so a reader never sees a half applied
      upstream change;
    * after a sync the overlay returns every row of the imported snapshot,
      including files appended after the index was built. Those files have no
      sidecar, so they are read from Parquet -- the fallback this issue asks
      for.

    This replaces the PyIceberg detached sync tests. PyIceberg support here is
    v2 only and is no longer used for Iceberg outside BenoDB, so it could not
    cover an external writer at all.
    """

    SCHEMA = "id BIGINT, title STRING, kind STRING"

    def _spark_write(self, warehouse: Path, action: str) -> None:
        """Run one Spark process against the upstream table.

        Two separate `spark-submit` invocations rather than one session: a JVM
        that starts after the mount is the closest local analogue of the
        external writer this issue is about (Databricks, Flink, a scheduled
        job), and it cannot have observed any BenoDB state.
        """
        import textwrap

        if action == "create":
            statement = (
                "spark.createDataFrame(\n"
                '    [(1, "alpha", "a"), (2, "beta", "b"), (3, "gamma", "a")],\n'
                f'    "{self.SCHEMA}",\n'
                ').writeTo("local.db.events").using("iceberg").create()\n'
            )
        elif action == "append":
            statement = (
                "spark.createDataFrame(\n"
                '    [(4, "delta", "b"), (5, "epsilon", "a")],\n'
                f'    "{self.SCHEMA}",\n'
                ').writeTo("local.db.events").append()\n'
            )
        else:
            raise AssertionError(f"unknown Spark write action: {action}")

        script = (
            textwrap.dedent(
                f"""\
                from pyspark.sql import SparkSession

                spark = (
                    SparkSession.builder
                    .appName("BenoDB Out Of Band Append")
                    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
                    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
                    .config("spark.sql.catalog.local.type", "hadoop")
                    .config("spark.sql.catalog.local.warehouse", "{warehouse}")
                    .getOrCreate()
                )
                """
            )
            + statement
        )

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            cmd = ["spark-submit", "--master", "local[*]"]
            if REQUIRE_JVM_INTEROP:
                cmd += _spark_runtime_args()
            cmd.append(script_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # The pip wheel ships no Iceberg runtime, so a local run without
                # the pinned jar fails at catalog construction rather than at an
                # assertion about rows. Java reports that as a plain
                # `ClassNotFoundException: org.apache.iceberg...`, not as the
                # `SparkClassNotFoundException` Spark uses for data sources.
                output = f"{result.stdout}\n{result.stderr}"
                if any(
                    marker in output
                    for marker in (
                        "ClassNotFoundException: org.apache.iceberg",
                        "NoClassDefFoundError: org/apache/iceberg",
                        "SparkClassNotFoundException",
                        "DATA_SOURCE_NOT_FOUND",
                    )
                ):
                    _require_or_skip("Iceberg Spark runtime JAR not available")
                assert result.returncode == 0, (
                    f"spark-submit failed:\n{result.stderr}\n{result.stdout}"
                )
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def test_spark_appends_out_of_band_between_builds(self, tmp_path):
        """Append upstream after the index is built, then reconcile."""
        if not _spark_submit_available():
            _require_or_skip("spark-submit not available")
        # The pip-installed pyspark carries no Iceberg runtime, so a per-PR shard
        # would start a JVM only to discover that. The interop workflow sets
        # BDB_REQUIRE_JVM_INTEROP=1 and supplies the pinned jar, which is where
        # this test is meant to run.
        if not REQUIRE_JVM_INTEROP and (
            os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
        ):
            pytest.skip("pyspark pip package lacks Iceberg JAR in CI")

        warehouse = tmp_path / "warehouse"
        overlay = tmp_path / "overlay"
        warehouse.mkdir()
        overlay.mkdir()

        self._spark_write(warehouse, "create")

        table = bdb.Table.mount_external_iceberg(
            table_uri=f"file://{warehouse}/db/events",
            overlay_storage_uri=f"file://{overlay}",
        )
        table.add_index("title", "bm25")
        table.wait_for_background_tasks()

        pinned = table.detached_catalog()["upstream_snapshot_id"]
        assert pinned is not None, "Spark committed a snapshot, so the mount must pin one"

        # The index serves the imported snapshot.
        assert sorted(table.sql("SELECT id FROM t").column("id").to_pylist()) == [1, 2, 3]
        plan = table.explain("title = 'alpha'")
        assert "Inverted Index" in plan or "BM25" in plan, (
            f"expected the BM25 index to serve the mounted snapshot:\n{plan}"
        )
        assert "access: Full Scan" not in plan, (
            f"expected index access, not a full scan:\n{plan}"
        )

        # Out of band: a second Spark process appends two rows.
        self._spark_write(warehouse, "append")

        # No sync yet. The overlay must stay pinned rather than adopt a snapshot
        # it has not imported, and must return that snapshot's rows.
        assert table.detached_catalog()["upstream_snapshot_id"] == pinned, (
            "the mount adopted an upstream snapshot it never imported"
        )
        assert sorted(table.sql("SELECT id FROM t").column("id").to_pylist()) == [1, 2, 3]

        # The sync imports the appended files. Nothing waits for the index build
        # here: the new files have no sidecar yet, which is the case the Parquet
        # fallback exists for, and all five rows must still come back.
        assert table.sync_detached_overlay() is True
        assert sorted(table.sql("SELECT id FROM t").column("id").to_pylist()) == [
            1,
            2,
            3,
            4,
            5,
        ], "rows from the appended file were missed after the sync"
        assert table.detached_catalog()["upstream_snapshot_id"] != pinned

        table.wait_for_background_tasks()

        # Reconciling did not drop the index that was already serving the first
        # file, and the appended row is readable by value, not just by count.
        plan = table.explain("title = 'alpha'")
        assert "Inverted Index" in plan or "BM25" in plan, (
            f"expected the BM25 index to survive the reconcile:\n{plan}"
        )
        assert "access: Full Scan" not in plan, (
            f"expected index access after the reconcile:\n{plan}"
        )
        row = table.sql("SELECT title FROM t WHERE id = 5")
        assert row.num_rows == 1
        assert row.column("title")[0].as_py() == "epsilon"


# Helper functions for manual testing
def create_sample_table_for_spark():
    """Create a sample table for manual Spark testing"""
    table_path = str(TEST_DIR / "manual_spark_test")
    table = bdb.Table(table_path)
    
    df = pd.DataFrame({
        "id": range(10000),
        "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
        "value": [i * 1.1 for i in range(10000)],
        "category": ["A", "B", "C", "D"] * 2500
    })
    
    table.set_sort_order(["timestamp"], ascending=[False])
    table.write_pandas(df)
    table.commit()
    
    print(f"✅ Created test table at: {table_path}")
    print("\nTo test with Spark:")
    print(f"  spark.read.format('iceberg').load('{table_path}').show()")
    
    return table_path


def create_sample_table_for_trino():
    """Create a sample table for manual Trino testing"""
    table_path = str(TEST_DIR / "manual_trino_test")
    table = bdb.Table(table_path)
    
    df = pd.DataFrame({
        "user_id": range(5000),
        "action": ["click", "view", "purchase", "return"] * 1250,
        "amount": [i * 2.5 for i in range(5000)]
    })
    
    table.write_pandas(df)
    table.commit()

    print(f"✅ Created test table at: {table_path}")
    print("\nTo test with Trino:")
    print(f"  SELECT COUNT(*) FROM iceberg.default.\"{table_path.split('/')[-1]}\";")
    
    return table_path


if __name__ == "__main__":
    print("Creating sample tables for manual testing...")
    print("\n" + "="*60)
    create_sample_table_for_spark()
    print("\n" + "="*60)
    create_sample_table_for_trino()
    print("\n" + "="*60)
    print("\n✅ Sample tables created!")
    print("\nRun pytest to execute automated tests:")
    print("  pytest tests/test_cross_engine_compat.py -v")
