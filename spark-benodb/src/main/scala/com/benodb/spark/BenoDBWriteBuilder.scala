package com.benodb.spark

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.StructType

/** Native write builder. */
class BenoDBWriteBuilder(table: BenoDBTable, info: LogicalWriteInfo, gpuDevice: String)
    extends WriteBuilder {
  override def build(): Write = new BenoDBWrite(table.tableUri, info.schema(), gpuDevice)
}

class BenoDBWrite(uri: String, writeSchema: StructType, gpuDevice: String) extends Write {
  /**
   * Resolved on the driver, before any task runs, so tasks and the commit agree
   * on the protocol. `files` is the Iceberg shape: tasks write data files, the
   * driver commits only metadata. `buffered` is used for tables that need
   * commit-time coordination (`format_version = 3` row ids).
   */
  private val mode: String = BenoDBArrowIpc.writeMode(uri)

  override def toBatch: BatchWrite =
    new BenoDBBatchWrite(uri, writeSchema, gpuDevice, mode)
}

/**
 * Commit message carrying one task's rows as Arrow IPC bytes.
 *
 * The task never touches the engine: it encodes and hands the bytes to the
 * driver, which appends them exactly once. Replaying a task attempt therefore
 * cannot duplicate rows.
 *
 * `Vector`, not `Seq`: on Scala 2.12 `Iterator.toSeq` yields a lazy `Stream`
 * whose tail captures the iterator, which Spark cannot serialize (`not
 * serializable result: scala.collection.Iterator$$anon$10`). A `Vector` is
 * strict and serializable on both 2.12 and 2.13.
 */
case class BenoDBCommitMessage(chunks: Vector[Array[Byte]]) extends WriterCommitMessage

/**
 * Commit message for the file protocol: the task's manifest entries, as JSON.
 *
 * Same shape as Iceberg's `SparkWrite$TaskCommit` — file metadata, not rows.
 */
case class BenoDBFileCommitMessage(entriesJson: String) extends WriterCommitMessage

class BenoDBBatchWrite(
    uri: String,
    writeSchema: StructType,
    gpuDevice: String,
    mode: String)
    extends BatchWrite {

  /**
   * Identifies *this* write attempt's staging buffer.
   *
   * Created once on the driver and carried to every task, so all commit
   * messages of the job land in the same buffer and are committed together.
   */
  private val token: String = java.util.UUID.randomUUID().toString

  override def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory =
    new BenoDBDataWriterFactory(uri, writeSchema, gpuDevice, mode)

  /**
   * Driver-side commit.
   *
   * `files`: the entries the tasks returned are committed in one metadata-only
   * snapshot — no data crosses the driver, and the commit is idempotent per token.
   * `buffered`: the rows are staged in the engine and written once (see
   * [[BenoDBCommitMessage]]).
   */
  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    if (mode == BenoDBArrowIpc.FilesMode) {
      val entries = messages.iterator
        .collect { case message: BenoDBFileCommitMessage => message.entriesJson }
        .toVector
      BenoDBArrowIpc.commitStagedEntries(uri, token, entries)
    } else {
      messages.foreach {
        case message: BenoDBCommitMessage =>
          message.chunks.foreach(chunk => BenoDBArrowIpc.stage(uri, token, chunk))
        case _ => ()
      }
      BenoDBArrowIpc.commitStaged(uri, token, gpuDevice)
    }
  }

  /**
   * Nothing is visible before the commit: buffered writes drop their staging
   * buffer, file writes leave their data files as orphans for
   * `remove_orphan_files`.
   */
  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    if (mode != BenoDBArrowIpc.FilesMode) {
      BenoDBArrowIpc.discard(token)
    }
  }
}

class BenoDBDataWriterFactory(
    uri: String,
    writeSchema: StructType,
    gpuDevice: String,
    mode: String)
    extends DataWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DataWriter[InternalRow] =
    new BenoDBDataWriter(uri, writeSchema, gpuDevice, mode)
}

class BenoDBDataWriter(
    uri: String,
    writeSchema: StructType,
    gpuDevice: String,
    mode: String)
    extends DataWriter[InternalRow] {

  private val batch = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private def copy(row: InternalRow): InternalRow =
    row.asInstanceOf[org.apache.spark.sql.catalyst.expressions.UnsafeRow].copy()

  override def write(record: InternalRow): Unit = batch += copy(record)

  override def commit(): WriterCommitMessage = {
    val chunks = batch
      .grouped(BenoDBDataWriter.ChunkRows)
      .map(chunk => BenoDBArrowIpc.encode(chunk.toSeq, writeSchema))
      .toVector
    batch.clear()
    if (mode == BenoDBArrowIpc.FilesMode) {
      BenoDBFileCommitMessage(BenoDBArrowIpc.stageDataFiles(uri, chunks))
    } else {
      BenoDBCommitMessage(chunks)
    }
  }

  override def abort(): Unit = batch.clear()
  override def close(): Unit = ()
}

object BenoDBDataWriter {
  /** Rows per encoded IPC chunk; bounds the transient encode buffer, not the task. */
  val ChunkRows: Int = 10000
}
