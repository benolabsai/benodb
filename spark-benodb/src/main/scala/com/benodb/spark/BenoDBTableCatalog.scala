package com.benodb.spark

import com.benodb.spark.jni.BenoDBJNIBridge
import org.apache.spark.sql.catalyst.analysis.{NoSuchNamespaceException, NoSuchTableException, TableAlreadyExistsException}
import org.apache.spark.sql.connector.catalog._
import org.apache.spark.sql.connector.catalog.functions.UnboundFunction
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util.{Map => JMap}
import scala.collection.JavaConverters._

/**
 * Native BenoDB catalog. Resolves/creates/drops tables entirely through
 * the engine's JNI metadata surface (`listSchemas`/`listTables`/`getTableSchema`
 * /`createTable`/`dropTable`) — no Iceberg catalog, so one Scala-2.13 artifact
 * serves Spark 4.0/4.1/4.2.
 *
 * Also implements the DSv2 `FunctionCatalog` so the vector/hybrid functions
 * resolve natively (`benostream.system.cosine_distance(...)` etc.) from SQL,
 * DataFrames, and PySpark with zero session registration.
 *
 * Configure like:
 * {{{
 *   spark.sql.catalog.benostream = com.benodb.spark.BenoDBCatalog
 *   spark.sql.catalog.benostream.warehouse = /path/to/warehouse   (or s3://bucket/prefix)
 * }}}
 */
class BenoDBCatalog extends TableCatalog with SupportsNamespaces with FunctionCatalog {

  private var catalogName: String = "benostream"
  private var warehouse: String = ""
  private var gpuDevice: String = "auto"

  private def bridge = BenoDBJNIBridge.getInstance()

  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {
    catalogName = name
    warehouse = Option(options.get("warehouse"))
      .orElse(Option(options.get("benostream.warehouse")))
      .getOrElse("")
      .stripSuffix("/")
    gpuDevice = options.getOrDefault("gpu_device", "auto")
  }

  override def name(): String = catalogName

  private def uriFor(namespace: Array[String], table: String): String =
    (warehouse +: namespace.toSeq :+ table).mkString("/")

  private def schemaOf(uri: String, ident: Identifier): StructType = {
    val json = bridge.getTableSchema(uri)
    if (json == null || json.isEmpty || json == "[]")
      throw new NoSuchTableException(ident)
    BenoDBArrowUtils.schemaJsonToStructType(json)
  }

  // ---- TableCatalog ----

  override def loadTable(ident: Identifier): Table = {
    val uri = uriFor(ident.namespace(), ident.name())
    new BenoDBTable(uri, schemaOf(uri, ident), new java.util.HashMap[String, String](), gpuDevice)
  }

  override def createTable(
      ident: Identifier,
      schema: StructType,
      partitions: Array[Transform],
      properties: JMap[String, String]
  ): Table = {
    val uri = uriFor(ident.namespace(), ident.name())
    if (!bridge.createTable(uri, BenoDBArrowUtils.structTypeToSchemaJson(schema)))
      throw new TableAlreadyExistsException(uri)
    new BenoDBTable(uri, schema, properties, gpuDevice)
  }

  override def alterTable(ident: Identifier, changes: TableChange*): Table =
    throw new UnsupportedOperationException("alterTable is not supported by the BenoDB catalog")

  override def dropTable(ident: Identifier): Boolean =
    bridge.dropTable(uriFor(ident.namespace(), ident.name()))

  override def renameTable(oldIdent: Identifier, newIdent: Identifier): Unit =
    throw new UnsupportedOperationException("renameTable is not supported by the BenoDB catalog")

  override def listTables(namespace: Array[String]): Array[Identifier] =
    BenoDBArrowUtils
      .parseStringArray(bridge.listTables(warehouse, namespace.lastOption.getOrElse("default")))
      .map(t => Identifier.of(namespace, t))
      .toArray

  // ---- SupportsNamespaces ----

  override def listNamespaces(): Array[Array[String]] =
    BenoDBArrowUtils.parseStringArray(bridge.listSchemas(warehouse)).map(s => Array(s)).toArray

  override def listNamespaces(namespace: Array[String]): Array[Array[String]] =
    if (namespace.isEmpty) listNamespaces() else Array.empty

  override def loadNamespaceMetadata(namespace: Array[String]): JMap[String, String] =
    new java.util.HashMap[String, String]()

  override def createNamespace(namespace: Array[String], metadata: JMap[String, String]): Unit =
    bridge.createSchema(warehouse, namespace.last)

  override def alterNamespace(namespace: Array[String], changes: NamespaceChange*): Unit =
    throw new UnsupportedOperationException("alterNamespace is not supported by the BenoDB catalog")

  override def dropNamespace(namespace: Array[String], cascade: Boolean): Boolean =
    throw new UnsupportedOperationException("dropNamespace is not supported by the BenoDB catalog")

  // ---- FunctionCatalog ----

  override def listFunctions(namespace: Array[String]): Array[Identifier] =
    if (namespace.length == 1 && namespace(0).equalsIgnoreCase("system"))
      BenoDBCatalogFunctions.names.map(n => Identifier.of(namespace, n)).toArray
    else Array.empty[Identifier]

  override def loadFunction(ident: Identifier): UnboundFunction = {
    val fnName = ident.name().toLowerCase
    if (!BenoDBCatalogFunctions.names.contains(fnName)) {
      throw new org.apache.spark.sql.catalyst.analysis.NoSuchFunctionException(ident)
    }
    BenoDBCatalogFunctions.unbound(ident)
  }
}
