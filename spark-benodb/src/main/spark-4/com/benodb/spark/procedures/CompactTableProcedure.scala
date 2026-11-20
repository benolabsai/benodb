package com.benodb.spark.procedures

import org.apache.spark.sql.connector.catalog.procedures.{BoundProcedure, ProcedureParameter, UnboundProcedure}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.Scan

class CompactTableProcedure extends UnboundProcedure with BoundProcedure {
  
  override def name(): String = "compact"
  override def description(): String = "Compacts table segments and consolidates indexes"
  override def isDeterministic: Boolean = true
  override def bind(inputType: StructType): BoundProcedure = this
  
  override def parameters(): Array[ProcedureParameter] = Array(
    ProcedureParameter.in("table", DataTypes.StringType).build()
  )
      
  override def call(inputArgs: InternalRow): java.util.Iterator[Scan] = {
    val table = inputArgs.getString(0)
    
    val jniBridge = com.benodb.spark.jni.BenoDBJNIBridge.getInstance()
    val gpuDevice = org.apache.spark.sql.SparkSession.active.conf.get("spark.benodb.gpu.device", "auto")
    jniBridge.setGpuContext(gpuDevice)
    val tableIdentifier = BenoDBTableResolver.resolveFromSession(table)
    jniBridge.compactTable(tableIdentifier)
    
    java.util.Collections.emptyIterator()
  }
}
