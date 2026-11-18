package com.benodb.trino;

import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ConnectorSession;

/**
 * Stored procedures exposed under {@code system} by the Trino connector.
 *
 * <p>Argument names are uppercase because Trino's {@code Procedure.Argument}
 * constructor enforces uppercase names for CALL compatibility.</p>
 *
 * <p>The parameter types here are {@code String}, not {@link io.airlift.slice.Slice},
 * and that asymmetry is deliberate. Two independent checks apply to a VARCHAR
 * argument:</p>
 * <ul>
 *   <li>{@code Procedure.Argument} validates a declared <em>default value</em>
 *       against {@code type.getJavaType()} (Slice for varchar), so the defaults
 *       in the connector factory must be {@code Slices.utf8Slice(...)}.</li>
 *   <li>the CALL execution path supplies {@code String} values and rejects a
 *       handle whose parameter type is Slice with "Argument 'TABLE' has invalid
 *       type io.airlift.slice.Slice (expected java.lang.String)".</li>
 * </ul>
 */
public final class BenoDBProcedures {

    private final String warehouse;

    public BenoDBProcedures(String warehouse) {
        this.warehouse = warehouse;
    }

    public void addIndex(ConnectorSession session, String table, String column, String algorithm) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        String resolvedAlgorithm = algorithm == null || algorithm.isBlank()
                ? "vector"
                : algorithm.trim();
        if (!BenoDBJNIBridge.addIndex(uri, column, resolvedAlgorithm)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB add_index failed for " + table + "." + column
                            + " using " + resolvedAlgorithm);
        }
    }

    public void buildIndex(ConnectorSession session, String table, String segmentId) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        String target = segmentId == null || segmentId.isBlank() ? "all" : segmentId.trim();
        if (!BenoDBJNIBridge.buildIndex(uri, target)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB build_index failed for " + table + " (" + target + ")");
        }
    }

    public void dropIndex(ConnectorSession session, String table, String column) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        if (!BenoDBJNIBridge.dropIndex(uri, column, "")) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB drop_index failed for " + table + "." + column);
        }
    }

    /**
     * Spark's {@code rebuild_index(table, column)} rebuilds the segment named by
     * the column, which is what the engine expects as a segment id.
     */
    public void rebuildIndex(ConnectorSession session, String table, String column) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        String target = column == null || column.isBlank() ? "all" : column.trim();
        if (!BenoDBJNIBridge.buildIndex(uri, target)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB rebuild_index failed for " + table + " (" + target + ")");
        }
    }

    public void compact(ConnectorSession session, String table) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        if (!BenoDBJNIBridge.compactTable(uri)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB compact failed for " + table);
        }
    }

    public void setPrimaryKey(ConnectorSession session, String table, String columns) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        if (!BenoDBJNIBridge.setPrimaryKey(uri, columns)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB set_primary_key failed for " + table + " (" + columns + ")");
        }
    }

    public void dropPrimaryKey(ConnectorSession session, String table, String columns) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        if (!BenoDBJNIBridge.dropPrimaryKey(uri, columns)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB drop_primary_key failed for " + table + " (" + columns + ")");
        }
    }

    public void rebuildClusterIndex(
            ConnectorSession session,
            String table,
            String column,
            String k,
            String m,
            String efConstruction,
            String targetShardMb) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        String col = column == null || column.isBlank() ? "embedding" : column.trim();
        int kVal = 1024;
        if (k != null && !k.isBlank()) {
            try {
                kVal = Integer.parseInt(k.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        int mVal = 16;
        if (m != null && !m.isBlank()) {
            try {
                mVal = Integer.parseInt(m.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        int efVal = 200;
        if (efConstruction != null && !efConstruction.isBlank()) {
            try {
                efVal = Integer.parseInt(efConstruction.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        long shardBytes = 536870912L;
        if (targetShardMb != null && !targetShardMb.isBlank()) {
            try {
                shardBytes = Long.parseLong(targetShardMb.trim()) * 1024L * 1024L;
            } catch (NumberFormatException ignored) {
            }
        }
        if (!BenoDBJNIBridge.rebuildClusterIndex(uri, col, kVal, mVal, efVal, shardBytes)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB rebuild_cluster_index failed for " + table + "." + col);
        }
    }

    public void warmup(
            ConnectorSession session,
            String table,
            String nProbes,
            String maxMemoryGb) {
        String uri = BenoDBTableUri.resolve(warehouse, table);
        int probes = 100;
        if (nProbes != null && !nProbes.isBlank()) {
            try {
                probes = Integer.parseInt(nProbes.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        double maxMem = 8.0;
        if (maxMemoryGb != null && !maxMemoryGb.isBlank()) {
            try {
                maxMem = Double.parseDouble(maxMemoryGb.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        if (!BenoDBJNIBridge.warmupTable(uri, probes, maxMem)) {
            throw new TrinoException(
                    StandardErrorCode.GENERIC_INTERNAL_ERROR,
                    "BenoDB warmup failed for " + table);
        }
    }
}
