package com.benodb.spark

import com.benodb.spark.procedures.BenoDBTableResolver
import org.junit.Assert._
import org.junit.Test

/**
 * Unit coverage for `BenoDBTableResolver`, the helper every stored procedure
 * uses to turn its `table` argument into an engine URI.
 *
 * The regression this guards against: `table.split(".").last` dropped the
 * namespace, so `benodb.default.users` and `users` were indistinguishable and
 * resolved against the wrong location.
 */
class BenoDBTableResolverTest {

  private val warehouse = "s3://bucket/warehouse/"

  @Test
  def bareTableResolvesToDefaultSchema(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/default/users",
      BenoDBTableResolver.resolve(warehouse, "users")
    )
  }

  @Test
  def schemaQualifiedTableKeepsTheNamespace(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/sales/users",
      BenoDBTableResolver.resolve(warehouse, "sales.users")
    )
  }

  @Test
  def catalogSchemaTableDropsOnlyTheCatalog(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/sales/users",
      BenoDBTableResolver.resolve(warehouse, "benodb.sales.users")
    )
  }

  @Test
  def explicitUriPassesThrough(): Unit = {
    val uri = "file:///tmp/warehouse/default/users"
    assertEquals(uri, BenoDBTableResolver.resolve(warehouse, uri))
  }

  @Test
  def whitespaceAndTrailingSlashesAreNormalized(): Unit = {
    assertEquals(
      "s3://bucket/warehouse/default/users",
      BenoDBTableResolver.resolve("  s3://bucket/warehouse//  ", "  users  ")
    )
  }

  @Test(expected = classOf[IllegalArgumentException])
  def emptyTableArgumentIsRejected(): Unit = {
    BenoDBTableResolver.resolve(warehouse, "   ")
  }

  @Test(expected = classOf[IllegalArgumentException])
  def emptyWarehouseIsRejected(): Unit = {
    BenoDBTableResolver.resolve("", "users")
  }

  @Test(expected = classOf[IllegalArgumentException])
  def tooManySegmentsAreRejected(): Unit = {
    BenoDBTableResolver.resolve(warehouse, "a.b.c.d")
  }
}
