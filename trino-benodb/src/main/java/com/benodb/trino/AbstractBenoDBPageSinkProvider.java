package com.benodb.trino;

import io.trino.spi.connector.ConnectorInsertTableHandle;
import io.trino.spi.connector.ConnectorMergeSink;
import io.trino.spi.connector.ConnectorMergeTableHandle;
import io.trino.spi.connector.ConnectorOutputTableHandle;
import io.trino.spi.connector.ConnectorPageSink;
import io.trino.spi.connector.ConnectorPageSinkId;
import io.trino.spi.connector.ConnectorPageSinkProvider;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.connector.ConnectorTableCredentials;
import io.trino.spi.connector.ConnectorTransactionHandle;

import java.util.Optional;

/**
 * Band-neutral sink provider.
 *
 * <p>The two {@code createPageSink} overloads are byte-identical across the
 * supported bands and live here. {@code createMergeSink} cannot: Trino 482-483
 * take five arguments, while 484+ append a {@code MemoryContext}. The
 * band-specific subclass supplies that one override and delegates to
 * {@link #createMergeSinkFor(ConnectorMergeTableHandle)}.</p>
 */
public abstract class AbstractBenoDBPageSinkProvider implements ConnectorPageSinkProvider {

    private final String warehouse;
    private final String gpuDevice;

    protected AbstractBenoDBPageSinkProvider(String warehouse, String gpuDevice) {
        this.warehouse = warehouse;
        this.gpuDevice = gpuDevice;
    }

    @Override
    public ConnectorPageSink createPageSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorOutputTableHandle outputTableHandle,
            Optional<ConnectorTableCredentials> credentials,
            ConnectorPageSinkId pageSinkId) {
        BenoDBOutputTableHandle out = (BenoDBOutputTableHandle) outputTableHandle;
        BenoDBInsertTableHandle insert = new BenoDBInsertTableHandle(
                out.getSchemaName(), out.getTableName(), out.getColumns());
        return new BenoDBPageSink(insert, warehouse, gpuDevice, true);
    }

    @Override
    public ConnectorPageSink createPageSink(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorInsertTableHandle insertTableHandle,
            Optional<ConnectorTableCredentials> credentials,
            ConnectorPageSinkId pageSinkId) {
        return new BenoDBPageSink((BenoDBInsertTableHandle) insertTableHandle, warehouse, gpuDevice);
    }

    /** Shared body of the band-specific {@code createMergeSink} overrides. */
    protected ConnectorMergeSink createMergeSinkFor(ConnectorMergeTableHandle mergeHandle) {
        return new BenoDBMergeSink((BenoDBMergeTableHandle) mergeHandle, warehouse, gpuDevice);
    }
}
