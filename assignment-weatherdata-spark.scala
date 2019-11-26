/**
 * By: William Henriksson, a15wilhe
 *
 * Assignment 1
 * BDP 2019
 */

import org.apache.spark.sql.types.{StructType, StructField, TimestampType, IntegerType, DoubleType, StringType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{window, col}

val schema = StructType(
  List(
    StructField("station", IntegerType, true)
    ,StructField("latitude", DoubleType, true)
    ,StructField("longitude", DoubleType, true)
    ,StructField("ts", TimestampType, true)
    ,StructField("value", DoubleType, true)
    ,StructField("quality", StringType, true)
    ,StructField("depth", DoubleType, true)
  )
)

/**
 *  read data
 *	Narrow transformations
 * 
 * The simple .schema did notwork so hade to work around it
 * Dropping last 2 columns
 */
// 
val dsDir = spark.
	read.format("csv").
	option("header", "true").
	option("delimiter", ";").
	load("C:/spark/workspace/smhi/direction/*").
	withColumn("depth", regexp_replace(col("depth"), "m", "")).
	drop("comment").drop("_c7").
	withColumn("station", $"station".cast(IntegerType)).
	withColumn("latitude", $"latitude".cast(DoubleType)).
	withColumn("longitude", $"longitude".cast(DoubleType)).
	withColumn("ts", $"ts".cast(TimestampType)).
	withColumn("value", $"value".cast(DoubleType)).
	withColumnRenamed("value", "direction").
	withColumn("quality", $"quality".cast(StringType)).
	withColumn("depth", $"depth".cast(DoubleType))

val dsFlux = spark.
	read.format("csv").
	option("header", "true").
	option("delimiter", ";").
	load("C:/spark/workspace/smhi/flux/*").
	withColumn("depth", regexp_replace(col("depth"), "m", "")).
	drop("comment").drop("_c7").
	withColumn("station", $"station".cast(IntegerType)).
	withColumn("latitude", $"latitude".cast(DoubleType)).
	withColumn("longitude", $"longitude".cast(DoubleType)).
	withColumn("ts", $"ts".cast(TimestampType)).
	withColumn("flux", $"value".cast(DoubleType)).
	withColumnRenamed("value", "flux").
	withColumn("quality", $"quality".cast(StringType)).
	withColumn("depth", $"depth".cast(DoubleType))

val dsSalt = spark.
	read.
	format("csv").
	option("header", "true").
	option("delimiter", ";").
	load("C:/spark/workspace/smhi/salinity/*").
	withColumn("depth", regexp_replace(col("depth"), "m", "")).
	drop("comment").drop("_c7").
	withColumn("station", $"station".cast(IntegerType)).
	withColumn("latitude", $"latitude".cast(DoubleType)).
	withColumn("longitude", $"longitude".cast(DoubleType)).
	withColumn("ts", $"ts".cast(TimestampType)).
	withColumn("value", $"value".cast(DoubleType)).
	withColumnRenamed("value", "salinity").
	withColumn("quality", $"quality".cast(StringType)).
	withColumn("depth", $"depth".cast(DoubleType))

val dsTemp = spark.
	read.
	format("csv").
	option("header", "true").
	option("delimiter", ";").
	load("C:/spark/workspace/smhi/temperature/*").
	withColumn("depth", regexp_replace(col("depth"), "m", "")).
	drop("comment").drop("_c7").
	withColumn("station", $"station".cast(IntegerType)).
	withColumn("latitude", $"latitude".cast(DoubleType)).
	withColumn("longitude", $"longitude".cast(DoubleType)).
	withColumn("ts", $"ts".cast(TimestampType)).
	withColumn("value", $"value".cast(DoubleType)).
	withColumnRenamed("value", "temp").
	withColumn("quality", $"quality".cast(StringType)).
	withColumn("depth", $"depth".cast(DoubleType))

/**
 * JOIN the 4 dataframes
 *
 * Join is a transformation that is most likely wide where shuffling takes place 
 * since the dataset is big
 *
 * joining on inner so i keep columns and null values to minimum
 * joining on col: station, lat, long and timestamp
 */

val join = dsDir.
	join(dsFlux, Seq("station", "latitude", "longitude", "ts")).
	join(dsSalt, Seq("station", "latitude", "longitude", "ts")).
	join(dsTemp, Seq("station", "latitude", "longitude", "ts"))


/**
 * first calculations
 * 
 */

// How many records are there? 
join.count 

// How many stations are there?
join
	.select($"station") // narrow
	.distinct // wide
	.count 

// How many recordings has each station made?
// Groupby stations to collect per station
join
	.groupBy(($"station")) //wide
	.count
	.show 

	//.collect.foreach(println)

/**
 * Aggregations on the join
 *
 * 
 * 
 */

// Mean temperature, salinity or flux (choose at least one) for all stations (not per station) over all years. 
// Per year, calc avg value and orderby year to get nice order
join
	.groupBy(year($"ts").as("year")) // wide
	.agg(avg($"salinity")) // wide when groupby taking place
	.orderBy($"year") //wide
	.show 

	//.collect.foreach(println)

// Mean temperature, salinity or flux (choose at least one) for all stations (not per station) for each month of each year, since year 2000. 
// first filter out all records older than 2000. Per each year and each month calc avg value and order 
join
	.filter(($"ts").gt(lit("1999-12-29"))) // narrow
	.groupBy(year($"ts").as("year"),month($"ts").as("month")) // wide
	.agg(avg($"salinity")) //wide when groupby taking place
	.orderBy($"year",$"month") //wide
	.show 

	//.collect.foreach(println)

// Mean temperature, salinity or flux (choose at least one) for all stations (not per station) for windows of 15 days, since year 2010. 
// filter out older than 2010, create a window of 15 days and groupby, 
// then calc avg value and order
join
	.filter(($"ts").gt(lit("2010-01-03"))) // narrow
	.groupBy(window(col("ts"), "15 days")) // wide
	.agg(avg($"salinity") as "mean") //wide when groupby taking place
	.orderBy("window.start")// wide
	.show(20, false)


//Mean temperature, salinity or flux (choose at least one) for all stations (not per station) for windows of 15 days with sliding windows of 3 days, since year 2010.
// filter out older than 2010, create a window of 15 days duration and sliding 
// window of 3 days and groupby, then calc avg value and order by window start to get nice order 
join
	.filter(($"ts").gt(lit("2010-01-03"))) // narrow
	.groupBy(window(col("ts"), "15 days", "3 days")) // wide
	.agg(avg($"salinity") as "mean") //wide when groupby taking place
	.orderBy("window.start") // wide
	.show(20, false)


//save point 3 (join) as parquet
join.write.format("parquet").save("C:/spark/workspace/salt")
join.write.format("parquet").saveAsTable("C:/spark/workspace/salt")