package com.benodb.spark

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.{Expressions, NamedReference}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Row-level operations (DELETE/UPDATE/MERGE) implemented over the engine's real
 * primitives: predicate deletes (`deleteRows`) and appends (`appendBatch`).
 *
 * Spark's merge-on-read needs a row identifier; the engine has no position
 * deletes, so we use the PRIMARY KEY columns as both `rowId` and the required
 * metadata. The PK is a normal data column, so it resolves on the target
 * relation and the scan already emits it. A delete/update therefore becomes a
 * `deleteRows(pk = ...)`; inserted/updated-new rows are appended.
 */
class BenoDBRowLevelOperationBuilder(table: BenoDBTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends RowLevelOperationBuilder {
  override def build(): RowLevelOperation = new BenoDBRowLevelOperation(table, info, gpuDevice)
}

class BenoDBRowLevelOperation(table: BenoDBTable, info: RowLevelOperationInfo, gpuDevice: String)
    extends SupportsDelta {

  private val pkColumns: Seq[String] = BenoDBRowLevelOperation.primaryKey(table)

  override def command(): RowLevelOperation.Command = info.command()
  override def description(): String = s"BenoDB native ${info.command()} on ${table.tableUri}"
  override def rowId(): Array[NamedReference] = pkColumns.map(Expressions.column).toArray
  override def requiredMetadataAttributes(): Array[NamedReference] = pkColumns.map(Expressions.column).toArray
  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = new BenoDBScanBuilder(table)
  override def newWriteBuilder(info: LogicalWriteInfo): DeltaWriteBuilder =
    new BenoDBDeltaWriteBuilder(
      table.tableUri,
      table.structSchema,
      pkColumns,
      gpuDevice,
      BenoDBDeltaWriter.resolveFlushSize(info, table.properties))
}

object BenoDBRowLevelOperation {
  def primaryKey(table: BenoDBTable): Seq[String] = {
    val declared = Option(table.properties.get("primary_key"))
      .map(_.split(",").map(_.trim).filter(_.nonEmpty).toSeq)
      .getOrElse(Seq.empty)
    if (declared.nonEmpty) declared
    else if (table.structSchema.nonEmpty) Seq(table.structSchema.fields.head.name)
    else Seq.empty
  }
}

class BenoDBDeltaWriteBuilder(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int = BenoDBDeltaWriter.DefaultFlushSize)
    extends DeltaWriteBuilder {
  override def build(): DeltaWrite = new BenoDBDeltaWrite(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

class BenoDBDeltaWrite(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaWrite {
  override def toBatch(): DeltaBatchWrite =
    new BenoDBDeltaBatchWrite(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

/**
 * One task's row-level work: the rendered delete predicates plus its inserts,
 * encoded as Arrow IPC. Nothing is applied until the driver commits, so a retried
 * task cannot delete or insert twice.
 *
 * `Vector`, not `Seq`: on Scala 2.12 `Iterator.toSeq` yields a lazy `Stream`
 * whose tail captures the iterator, which Spark cannot serialize.
 */
case class BenoDBDeltaCommitMessage(predicates: Vector[String], chunks: Vector[Array[Byte]])
    extends WriterCommitMessage

class BenoDBDeltaBatchWrite(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaBatchWrite {

  /** Identifies this write attempt's staging buffer (driver-generated). */
  private val token: String = java.util.UUID.randomUUID().toString

  override def createBatchWriterFactory(info: PhysicalWriteInfo): DeltaWriterFactory =
    new BenoDBDeltaWriterFactory(uri, dataSchema, pkColumns, gpuDevice, flushSize)

  /**
   * Driver-side commit: all deletes in one commit, then one snapshot of inserts.
   *
   * Deletes go first (an UPDATE reuses the key, so appending first would let the
   * delete reap the freshly inserted row). Both phases are driven from the
   * driver, and the staging buffer is drained before it is written, so a retried
   * task or a retried commit cannot apply the work twice.
   */
  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    com.benodb.spark.gpu.GpuContextResolver.bindTaskGpuContext(gpuDevice)
    val predicates = messages.iterator
      .collect { case message: BenoDBDeltaCommitMessage => message.predicates }
      .flatten
      .toVector
    BenoDBArrowIpc.deleteRows(uri, predicates)
    messages.foreach {
      case message: BenoDBDeltaCommitMessage =>
        message.chunks.foreach(chunk => BenoDBArrowIpc.stage(uri, token, chunk))
      case _ => ()
    }
    BenoDBArrowIpc.commitStaged(uri, token, gpuDevice)
  }

  /** Nothing was written before the commit, so there is nothing to undo. */
  override def abort(messages: Array[WriterCommitMessage]): Unit =
    BenoDBArrowIpc.discard(token)
}

class BenoDBDeltaWriterFactory(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int)
    extends DeltaWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DeltaWriter[InternalRow] =
    new BenoDBDeltaWriter(uri, dataSchema, pkColumns, gpuDevice, flushSize)
}

object BenoDBDeltaWriter {
  /** Rows/keys buffered before an intermediate delete or append is issued. */
  val DefaultFlushSize: Int = 10000

  /** Keys per `pk IN (...)` predicate for a single-column key. */
  val MaxKeysPerInList: Int = 1000

  /** Predicates per `(a = .. AND b = ..) OR ...` chunk for compound keys. */
  val MaxCompoundKeysPerChunk: Int = 500

  val BatchSizeKey: String = "write.batchSize"
  val BatchSizeConf: String = "spark.benostream.write.batchSize"

  /** Resolve the flush size: write option → table property → Spark conf → default. */
  def resolveFlushSize(info: LogicalWriteInfo, properties: java.util.Map[String, String]): Int = {
    def parse(raw: String): Option[Int] = scala.util.Try(raw.trim.toInt).toOption.filter(_ > 0)
    Option(info.options().get(BatchSizeKey))
      .flatMap(parse)
      .orElse(Option(properties.get(BatchSizeKey)).flatMap(parse))
      .orElse(
        org.apache.spark.sql.SparkSession.getActiveSession
          .flatMap(s => parse(s.conf.get(BatchSizeConf, ""))))
      .getOrElse(DefaultFlushSize)
  }
}

/**
 * Buffers row-level operations; the driver applies them in bounded chunks.
 *
 * Deletes accumulate *rendered primary-key literals*, not one giant SQL string:
 * a 10k-row DELETE used to build a multi-megabyte `pk = .. OR pk = ..` predicate
 * that either hit the engine's AST recursion limit or exhausted memory. Keys are
 * rendered as (a) `pk IN (lit, ..)` batches of
 * [[BenoDBDeltaWriter.MaxKeysPerInList]] for single-column keys, or
 * (b) `(a = .. AND b = ..) OR ...` chunks of
 * [[BenoDBDeltaWriter.MaxCompoundKeysPerChunk]] for compound keys. Rows whose
 * key contains a NULL cannot join an `IN` list and get their own
 * `pk IS NULL`-style predicate.
 *
 * Nothing is mutated from the task: the delete predicates and the inserts travel
 * to the driver in the commit message, so a retried task attempt cannot delete or
 * insert twice. Inserts are encoded in `flushSize`-row chunks to bound the
 * transient encode buffer.
 */
class BenoDBDeltaWriter(
    uri: String,
    dataSchema: StructType,
    pkColumns: Seq[String],
    gpuDevice: String,
    flushSize: Int = BenoDBDeltaWriter.DefaultFlushSize)
    extends DeltaWriter[InternalRow] {

  import BenoDBDeltaWriter._

  private val pkTypes = pkColumns.map(c => dataSchema(c).dataType)

  /** Rendered PK literals per pending delete, in `pkColumns` order. */
  private val deleteKeys = scala.collection.mutable.ArrayBuffer.empty[Seq[String]]

  /** Pending deletes whose key has a NULL component (cannot go in an `IN` list). */
  private val nullKeyPredicates = scala.collection.mutable.ArrayBuffer.empty[String]

  private val inserts = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private val effectiveFlushSize: Int = if (flushSize > 0) flushSize else DefaultFlushSize
  private val keysPerChunk: Int =
    if (pkColumns.length <= 1) MaxKeysPerInList else MaxCompoundKeysPerChunk

  private def quote(s: String): String = "\"" + s.replace("\"", "\"\"") + "\""
  private def literal(v: Any, isString: Boolean): String =
    if (v == null) "NULL" else if (isString) "'" + v.toString.replace("'", "''") + "'" else v.toString

  // metadata/id row = PK columns in `pkColumns` order
  private def recordDelete(row: InternalRow): Unit = {
    // No primary key means no predicate we can build; a table without a key
    // cannot service UPDATE/DELETE through the predicate-delete primitive, and
    // rendering an empty key list would index `pkColumns.head` on an empty Seq.
    if (pkColumns.isEmpty || row == null || row.numFields < pkColumns.length) return
    val keys = (0 until pkColumns.length).map { i =>
      if (row.isNullAt(i)) null
      else literal(row.get(i, pkTypes(i)), BenoDBSql.isStringType(pkTypes(i)))
    }
    if (keys.contains(null)) {
      nullKeyPredicates += pkColumns.indices
        .map(i => if (keys(i) == null) s"${quote(pkColumns(i))} IS NULL" else s"${quote(pkColumns(i))} = ${keys(i)}")
        .mkString(" AND ")
    } else {
      deleteKeys += keys
    }
    // No engine call and no flush here: buffering the whole partition is what
    // makes the driver-side commit exactly-once.
  }

  /**
   * Render the buffered keys as chunked predicates.
   *
   * Pure string work: the engine call happens in the driver-side commit so a
   * retried task attempt cannot delete the same rows twice.
   */
  private def renderDeletePredicates(): Seq[String] = {
    if (nullKeyPredicates.isEmpty && deleteKeys.isEmpty) return Seq.empty

    val predicates = scala.collection.mutable.ArrayBuffer.empty[String]
    predicates ++= nullKeyPredicates
    deleteKeys.iterator
      .grouped(keysPerChunk)
      .foreach(chunk => predicates += renderPredicate(chunk.map(_.toSeq).toSeq))
    nullKeyPredicates.clear()
    deleteKeys.clear()
    predicates.toSeq
  }

  private def renderPredicate(chunk: Seq[Seq[String]]): String =
    if (pkColumns.length <= 1) {
      s"${quote(pkColumns.head)} IN (${chunk.map(_.head).mkString(", ")})"
    } else {
      chunk
        .map(keys => pkColumns.zip(keys).map { case (column, value) => s"${quote(column)} = $value" }.mkString("(", " AND ", ")"))
        .mkString(" OR ")
    }

  override def delete(metadata: InternalRow, id: InternalRow): Unit =
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)

  override def update(metadata: InternalRow, id: InternalRow, row: InternalRow): Unit = {
    recordDelete(if (id != null && id.numFields >= pkColumns.length) id else metadata)
    inserts += row.copy()
  }

  override def insert(row: InternalRow): Unit =
    inserts += row.copy()

  override def commit(): WriterCommitMessage = {
    // DELETE before INSERT: an UPDATE reuses the key, so appending first would
    // let the delete reap the freshly inserted row. Both are applied by the
    // driver in `BenoDBDeltaBatchWrite.commit`.
    val predicates = renderDeletePredicates().toVector
    val chunks = inserts
      .grouped(effectiveFlushSize)
      .map(chunk => BenoDBArrowIpc.encode(chunk.toSeq, dataSchema))
      .toVector
    inserts.clear()
    BenoDBDeltaCommitMessage(predicates, chunks)
  }

  override def abort(): Unit = {
    deleteKeys.clear()
    nullKeyPredicates.clear()
    inserts.clear()
  }

  override def close(): Unit = {
    deleteKeys.clear()
    nullKeyPredicates.clear()
    inserts.clear()
  }
}
