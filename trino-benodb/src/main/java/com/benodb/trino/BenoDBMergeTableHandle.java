package com.benodb.trino;

import io.trino.spi.connector.ConnectorMergeTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Handle for a MERGE into a BenoDB table. Wraps the target table handle,
 * the insert handle used for the merged rows, and the hidden row-id column
 * (whose type follows the primary key, so non-numeric keys work).
 */
public class BenoDBMergeTableHandle implements ConnectorMergeTableHandle {
    private final BenoDBTableHandle tableHandle;
    private final BenoDBInsertTableHandle insertHandle;
    private final BenoDBColumnHandle rowIdColumn;

    @JsonCreator
    public BenoDBMergeTableHandle(
            @JsonProperty("tableHandle") BenoDBTableHandle tableHandle,
            @JsonProperty("insertHandle") BenoDBInsertTableHandle insertHandle,
            @JsonProperty("rowIdColumn") BenoDBColumnHandle rowIdColumn) {
        this.tableHandle = tableHandle;
        this.insertHandle = insertHandle;
        this.rowIdColumn = rowIdColumn;
    }

    @Override
    @JsonProperty
    public BenoDBTableHandle getTableHandle() {
        return tableHandle;
    }

    @JsonProperty
    public BenoDBInsertTableHandle getInsertHandle() {
        return insertHandle;
    }

    @JsonProperty
    public BenoDBColumnHandle getRowIdColumn() {
        return rowIdColumn;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        BenoDBMergeTableHandle that = (BenoDBMergeTableHandle) o;
        return Objects.equals(tableHandle, that.tableHandle)
                && Objects.equals(insertHandle, that.insertHandle)
                && Objects.equals(rowIdColumn, that.rowIdColumn);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tableHandle, insertHandle, rowIdColumn);
    }
}
