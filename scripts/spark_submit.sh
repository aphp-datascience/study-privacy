#!/bin/bash
set -x 
set -e
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
export SPARK_HOME=/usr/hdp/current/spark-2.4.3-client


$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode client \
--queue default \
--conf spark.driver.memory=8g \
--conf spark.yarn.am.memory=1g \
--conf spark.sql.autoBroadcastJoinThreshold=-1 \
--jars /opt/lib/EdsTools-0.0.1-SNAPSHOT.jar,/opt/lib/delta-core_2.11-0.5.1-SNAPSHOT.jar,/opt/lib/spark-avro_2.11-2.4.1.jar \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.ui.enabled=false \
--conf spark.kryoserializer.buffer.max=2047m \
--conf spark.driver.port=20107 \
--conf spark.driver.blockManager.port=20107 \
--conf spark.driver.bindAddress=0.0.0.0 \
--conf spark.driver.host=spark-client.jupyterhub.eds-int.aphp.fr \
--conf spark.home=/usr/hdp/current/spark-2.4.3-client \
--conf spark.local.dir=/tmp/spark \
--conf spark.yarn.appMasterEnv.HADOOP_USER_NAME=$USER \
--conf spark.driver.extraJavaOptions="-Dhdp.version=2.6.5.0-292 -Dhttp.proxyHost=10.143.10.20 -Dhttp.proxyPort=8080 -Dhttps.proxyHost=10.143.10.20 -Dhttps.proxyPort=8080 -Djava.security.auth.login.config=/export/home/$USER/.jaas.conf" \
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