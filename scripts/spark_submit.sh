#!/bin/bash
set -x 
set -e
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
export SPARK_HOME=/usr/hdp/current/spark-2.4.3-client
export HADOOP_VERSION=2.6.5.0-292


$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode client \
--queue default \
--conf spark.driver.memory=8g \
--conf spark.yarn.am.memory=1g \
--conf spark.sql.autoBroadcastJoinThreshold=-1 \
--jars $JARS_PATH/delta-core_2.11-0.5.1-SNAPSHOT.jar,$JARS_PATH/spark-avro_2.11-2.4.1.jar \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.ui.enabled=false \
--conf spark.kryoserializer.buffer.max=2047m \
--conf spark.driver.port=$SPARK_PORT \
--conf spark.driver.blockManager.port=$SPARK_PORT \
--conf spark.driver.bindAddress=0.0.0.0 \
--conf spark.driver.host=$SPARK_HOST \
--conf spark.home=$SPARK_HOME \
--conf spark.local.dir=/tmp/spark \
--conf spark.yarn.appMasterEnv.HADOOP_USER_NAME=$USER \
--conf spark.executor.cores=5 \
--conf spark.executor.memory=4g \
--conf spark.dynamicAllocation.maxExecutors=16 \
--conf spark.dynamicAllocation.minExecutors=4 \
--conf spark.default.parallelism=200 \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.io.compression.codec=lzf \
--conf spark.sql.hive.convertMetastoreOrc=false \
--conf spark.sql.session.timeZone=UTC \
--conf spark.sql.orc.enabled=true \
--conf spark.shuffle.service.enabled=true \
--conf spark.sql.shuffle.partitions=200 \
--conf spark.executorEnv.HOME=$HOME \
--conf spark.eventLog.enabled=true \
--conf spark.pyspark.python=$PYSPARK_PYTHON \
$1 $2 $3 $4