package com.benodb.trino;

import io.trino.spi.connector.ColumnHandle;
import io.trino.spi.connector.ConnectorSplit;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.connector.SchemaTableName;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.VarcharType;
import org.junit.Assume;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Unit tests for the Trino BenoDB connector components.
 *
 * Tests that do not need the native engine run unconditionally. Tests that
 * exercise metadata/query paths are gated on {@link BenoDBJNIBridge#isLoaded()}
 * so the suite still runs on a machine without {@code libbenodb}.
 */
public class TrinoConnectorTest {

    // ---- Plugin Tests ----

    @Test
    public void testPluginReturnsConnectorFactory() {
        BenoDBPlugin plugin = new BenoDBPlugin();
        var factories = plugin.getConnectorFactories();

        assertNotNull("Connector factories should not be null", factories);
        int count = 0;
        for (io.trino.spi.connector.ConnectorFactory factory : factories) {
            count++;
            assertTrue("Factory should be BenoDBConnectorFactory",
                    factory instanceof BenoDBConnectorFactory);
        }
        assertEquals("Should have exactly one factory", 1, count);
    }

    // ---- Handle equality (required by Trino's planner) ----

    @Test
    public void testColumnHandleValueEquality() {
        // Trino's MERGE planner compares ColumnHandles across separate
        // getColumnHandles() calls via List.indexOf (QueryPlanner.planMerge), so
        // value equality is mandatory. Two independently constructed handles for
        // the same column must be equal and share a hashCode.
        BenoDBColumnHandle a = new BenoDBColumnHandle("id", IntegerType.INTEGER);
        BenoDBColumnHandle b = new BenoDBColumnHandle("id", IntegerType.INTEGER);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, new BenoDBColumnHandle("value", IntegerType.INTEGER));
        assertNotEquals(a, new BenoDBColumnHandle("id", VarcharType.VARCHAR));

        // The hidden row-id handle must also compare by value.
        assertEquals(BenoDBColumnHandle.rowId("id"), BenoDBColumnHandle.rowId("id"));
        assertNotEquals(BenoDBColumnHandle.rowId("id"), BenoDBColumnHandle.rowId("other"));
    }

    @Test
    public void testTableHandleValueEquality() {
        assertEquals(new BenoDBTableHandle("s", "t"), new BenoDBTableHandle("s", "t"));
        assertNotEquals(new BenoDBTableHandle("s", "t"), new BenoDBTableHandle("s", "u"));
    }

    @Test
    public void testSplitValueEquality() {
        assertEquals(new BenoDBSplit("uri", "SELECT 1"), new BenoDBSplit("uri", "SELECT 1"));
        assertNotEquals(new BenoDBSplit("uri", "SELECT 1"), new BenoDBSplit("uri", "SELECT 2"));
    }

    // ---- ConnectorFactory Tests ----

    @Test
    public void testConnectorFactoryName() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        assertEquals("Connector name should be benodb", "benodb", factory.getName());
    }

    @Test
    public void testConnectorFactoryCreate() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test_catalog", Map.of(), null);

        assertNotNull("Connector should not be null", connector);
    }

    @Test
    public void testConnectorGetMetadata() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var metadata = connector.getMetadata(null, null);
        assertNotNull("Metadata should not be null", metadata);
        assertTrue("Should be BenoDBMetadata", metadata instanceof BenoDBMetadata);
    }

    @Test
    public void testConnectorGetSplitManager() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var splitManager = connector.getSplitManager();
        assertNotNull("Split manager should not be null", splitManager);
        assertTrue("Should be BenoDBSplitManager", splitManager instanceof BenoDBSplitManager);
    }

    @Test
    public void testConnectorGetPageSourceProvider() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var provider = connector.getPageSourceProvider();
        assertNotNull("Page source provider should not be null", provider);
        assertTrue("Should be BenoDBPageSourceProvider", provider instanceof BenoDBPageSourceProvider);
    }

    @Test
    public void testConnectorBeginTransaction() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var tx = connector.beginTransaction(
                io.trino.spi.transaction.IsolationLevel.SERIALIZABLE, true, true);
        assertNotNull("Transaction handle should not be null", tx);
        assertTrue("Should be BenoDBTransactionHandle",
                tx instanceof BenoDBConnectorFactory.BenoDBTransactionHandle);
    }

    // ---- Metadata: no mock fallback ----

    @Test
    public void testMetadataFailsLoudlyWithoutNative() {
        // The connector must never silently return mock metadata. When the
        // native library is absent, metadata calls throw.
        Assume.assumeFalse("native library is loaded; skip the no-native assertion",
                BenoDBJNIBridge.isLoaded());

        BenoDBMetadata metadata = new BenoDBMetadata();
        try {
            metadata.listSchemaNames(null);
            fail("listSchemaNames should throw when the native library is unavailable");
        } catch (IllegalStateException expected) {
            // expected
        }
    }

    @Test
    public void testMetadataListSchemaNames() {
        Assume.assumeTrue(BenoDBJNIBridge.isLoaded());
        BenoDBMetadata metadata = new BenoDBMetadata();
        var schemas = metadata.listSchemaNames(null);
        assertNotNull("Schema list should not be null", schemas);
    }

    @Test
    public void testMetadataListTables() {
        Assume.assumeTrue(BenoDBJNIBridge.isLoaded());
        BenoDBMetadata metadata = new BenoDBMetadata();
        var tables = metadata.listTables(null, java.util.Optional.empty());
        assertNotNull("Table list should not be null", tables);
    }

    // ---- TableHandle Tests ----

    @Test
    public void testTableHandleGetters() {
        BenoDBTableHandle handle = new BenoDBTableHandle("public", "users");

        assertEquals("Schema name should match", "public", handle.getSchemaName());
        assertEquals("Table name should match", "users", handle.getTableName());
    }

    // ---- ColumnHandle Tests ----

    @Test
    public void testColumnHandleGetters() {
        BenoDBColumnHandle handle = new BenoDBColumnHandle("email", VarcharType.VARCHAR);

        assertEquals("Column name should match", "email", handle.getColumnName());
        assertEquals("Column type should match", VarcharType.VARCHAR, handle.getColumnType());
    }

    @Test
    public void testColumnHandleImmutability() {
        BenoDBColumnHandle handle = new BenoDBColumnHandle("id", IntegerType.INTEGER);

        assertEquals("id", handle.getColumnName());
        assertEquals(IntegerType.INTEGER, handle.getColumnType());
    }

    // ---- Split Tests ----

    @Test
    public void testSplitConstruction() {
        BenoDBSplit split = new BenoDBSplit(
                "s3://bucket/table", "SELECT * FROM t WHERE id = 1");

        assertEquals("Table URI should match", "s3://bucket/table", split.getTableUri());
        assertEquals("SQL should match", "SELECT * FROM t WHERE id = 1", split.getSql());
    }

    @Test
    public void testSplitIsRemotelyAccessible() {
        BenoDBSplit split = new BenoDBSplit("s3://bucket/table", "SELECT * FROM t");
        assertTrue("Split should be remotely accessible", split.isRemotelyAccessible());
    }

    @Test
    public void testSplitGetAddresses() {
        BenoDBSplit split = new BenoDBSplit("s3://bucket/table", "SELECT * FROM t");
        var addresses = split.getAddresses();
        assertNotNull("Addresses should not be null", addresses);
        assertTrue("Addresses should be empty (managed by connector)", addresses.isEmpty());
    }

    // ---- SplitManager Tests ----

    // NOTE: there is deliberately no getSplitInfo() test. ConnectorSplit now
    // exposes no split-info accessor in either supported band (468's
    // getSplitInfo was removed by 482), so asserting on it would only pin a
    // version we no longer ship.

    @Test
    public void testSplitManagerEmitsSingleSplitWithPushedDownSql() throws Exception {
        BenoDBSplitManager splitManager = new BenoDBSplitManager(
                "s3://my-bucket/wh", "auto");
        BenoDBTableHandle tableHandle = new BenoDBTableHandle(
                "default", "events", java.util.Optional.of("severity = 'ERROR'"));

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        assertNotNull("Split source should not be null", splitSource);

        // getNextBatch takes (maxSize, DynamicFilterSnapshot) in both supported
        // bands. FixedSplitSource never consults the snapshot, so null is fine.
        List<ConnectorSplit> splits = splitSource.getNextBatch(1000, null).get();
        assertEquals("Should emit exactly one split", 1, splits.size());

        BenoDBSplit split = (BenoDBSplit) splits.get(0);
        assertEquals("s3://my-bucket/wh/default/events", split.getTableUri());
        assertEquals("SELECT * FROM t WHERE severity = 'ERROR'", split.getSql());
    }

    @Test
    public void testSplitManagerWithoutFilter() throws Exception {
        BenoDBSplitManager splitManager = new BenoDBSplitManager(
                "s3://my-bucket/wh", "auto");
        BenoDBTableHandle tableHandle = new BenoDBTableHandle("default", "events");

        var splitSource = splitManager.getSplits(null, null, tableHandle, null, null);
        List<ConnectorSplit> splits = splitSource.getNextBatch(1000, null).get();
        assertEquals("SELECT * FROM t", ((BenoDBSplit) splits.get(0)).getSql());
    }

    // ---- Write / MERGE path tests ----

    @Test
    public void testConnectorGetPageSinkProvider() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);

        var provider = connector.getPageSinkProvider();
        assertNotNull("Page sink provider should not be null", provider);
        assertTrue("Should be BenoDBPageSinkProvider",
                provider instanceof BenoDBPageSinkProvider);
    }

    @Test
    public void testConnectorFactoryProcedures() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test", Map.of(), null);
        var procedures = connector.getProcedures();
        assertNotNull("Procedures should not be null", procedures);

        var names = procedures.stream().map(io.trino.spi.procedure.Procedure::getName).toList();
        assertTrue("Should include rebuild_cluster_index", names.contains("rebuild_cluster_index"));
        assertTrue("Should include warmup", names.contains("warmup"));
        assertTrue("Should include add_index", names.contains("add_index"));
        assertTrue("Should include compact", names.contains("compact"));
    }

    @Test
    public void testBeginInsertReturnsHandle() {
        BenoDBMetadata metadata = new BenoDBMetadata();
        BenoDBTableHandle table = new BenoDBTableHandle("default", "test_table");
        List<ColumnHandle> columns = List.of(
                new BenoDBColumnHandle("id", IntegerType.INTEGER),
                new BenoDBColumnHandle("name", VarcharType.VARCHAR));

        var handle = metadata.beginInsert(null, table, columns,
                io.trino.spi.connector.RetryMode.NO_RETRIES);
        assertTrue("Should be BenoDBInsertTableHandle",
                handle instanceof BenoDBInsertTableHandle);

        BenoDBInsertTableHandle insert = (BenoDBInsertTableHandle) handle;
        assertEquals("default", insert.getSchemaName());
        assertEquals("test_table", insert.getTableName());
        assertEquals("Should carry the insert columns", 2, insert.getColumns().size());
    }

    @Test
    public void testArrowConverterSchemaMapping() {
        List<BenoDBColumnHandle> columns = List.of(
                new BenoDBColumnHandle("id", IntegerType.INTEGER),
                new BenoDBColumnHandle("name", VarcharType.VARCHAR));

        var schema = BenoDBArrowConverter.toArrowSchema(columns);
        assertEquals("Should have 2 fields", 2, schema.getFields().size());
        assertEquals("id", schema.getFields().get(0).getName());
        assertEquals("name", schema.getFields().get(1).getName());
    }

    @Test
    public void testTableUriMapping() {
        assertEquals("s3://default/default/test_table",
                BenoDBTableUri.of(BenoDBTableUri.DEFAULT_WAREHOUSE, "default", "test_table"));
    }

    @Test
    public void testTableUriMappingCustomWarehouse() {
        assertEquals("s3://my-bucket/warehouse/default/test_table",
                BenoDBTableUri.of("s3://my-bucket/warehouse", "default", "test_table"));
        // A trailing slash is normalized.
        assertEquals("s3://my-bucket/warehouse/default/test_table",
                BenoDBTableUri.of("s3://my-bucket/warehouse/", "default", "test_table"));
    }

    @Test
    public void testConnectorFactoryWarehouseConfig() {
        BenoDBConnectorFactory factory = new BenoDBConnectorFactory();
        var connector = factory.create("test",
                Map.of("benostream.warehouse", "s3://my-bucket/wh"), null);
        assertNotNull("Connector should not be null", connector);

        var metadata = connector.getMetadata(null, null);
        assertTrue("Should be BenoDBMetadata", metadata instanceof BenoDBMetadata);
    }

    @Test
    public void testBeginCreateTableReturnsHandle() {
        BenoDBMetadata metadata = new BenoDBMetadata();
        var tableMetadata = new io.trino.spi.connector.ConnectorTableMetadata(
                new SchemaTableName("default", "new_table"),
                List.of(
                        new io.trino.spi.connector.ColumnMetadata("id", IntegerType.INTEGER),
                        new io.trino.spi.connector.ColumnMetadata("name", VarcharType.VARCHAR)));

        var handle = metadata.beginCreateTable(null, tableMetadata, java.util.Optional.empty(),
                io.trino.spi.connector.RetryMode.NO_RETRIES, false);
        assertTrue("Should be BenoDBOutputTableHandle",
                handle instanceof BenoDBOutputTableHandle);

        BenoDBOutputTableHandle out = (BenoDBOutputTableHandle) handle;
        assertEquals("default", out.getSchemaName());
        assertEquals("new_table", out.getTableName());
        assertEquals("Should carry the output columns", 2, out.getColumns().size());
    }

    @Test
    public void testSchemaJsonMapping() {
        List<BenoDBColumnHandle> columns = List.of(
                new BenoDBColumnHandle("id", IntegerType.INTEGER),
                new BenoDBColumnHandle("name", VarcharType.VARCHAR));

        String json = BenoDBArrowConverter.toSchemaJson(columns);
        assertTrue("Should contain the id field", json.contains("\"name\":\"id\""));
        assertTrue("Should map INTEGER to Int32", json.contains("\"type\":\"Int32\""));
        assertTrue("Should contain the name field", json.contains("\"name\":\"name\""));
        assertTrue("Should map VARCHAR to Utf8", json.contains("\"type\":\"Utf8\""));
    }
}
