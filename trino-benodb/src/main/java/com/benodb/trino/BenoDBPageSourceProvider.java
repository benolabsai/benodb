package com.benodb.trino;

import io.trino.spi.connector.*;
import java.util.List;
import java.util.Optional;

public class BenoDBPageSourceProvider implements ConnectorPageSourceProvider {
    private final String gpuDevice;

    public BenoDBPageSourceProvider(String gpuDevice) {
        this.gpuDevice = gpuDevice;
    }

    @Override
    public ConnectorPageSource createPageSource(
            ConnectorTransactionHandle transaction,
            ConnectorSession session,
            ConnectorSplit split,
            ConnectorTableHandle table,
            Optional<ConnectorTableCredentials> credentials,
            List<ColumnHandle> columns,
            DynamicFilter dynamicFilter) {

        if (split instanceof ListIndexesSplit listSplit) {
            return new ListIndexesPageSource(listSplit.getJson());
        }

        if (split instanceof RegionalDriftFunction.Split driftSplit) {
            return new RegionalDriftPageSource(driftSplit);
        }

        BenoDBSplit hSplit = (BenoDBSplit) split;
        return new BenoDBPageSource(hSplit, columns, gpuDevice);
    }
}
