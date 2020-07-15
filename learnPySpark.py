# coding:utf-8
# python3

"""
spark2-submit \
--master yarn \
--deploy-mode client \
--executor-memory 2G \
--num-executors 10 \
--executor-cores 4 \
--conf spark.pyspark.python=/home/work/app/python3/bin/python3 \
/home/work/remote/code/script/tmp/learnPySpark.py
"""

import os
import socket

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext


def environ_init():
    hostname = socket.gethostname()
    if "hadoop" not in hostname:
        # os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk1.8.0_221"
        os.environ["HADOOP_HOME"] = "E:/Appilaction/hadoop-common-2.6.0-bin"
        os.environ["SPARK_HOME"] = "E:/Appilaction/spark-2.4.4-bin-hadoop2.6"

        # 使用spark2-submit提交任务需要指定--conf spark.pyspark.python=/home/work/app/python3/bin/python3
        os.environ["PYSPARK_PYTHON"] = "C:/Program Files/Python36/python.exe"

        # org.apache.hadoop.security.AccessControlException: Permission denied: user=Admin, access=READ_EXECUTE, inode="/user/hive/warehouse/medical.db/dim_date":hive:hive:drwxrwx--x
        os.environ["HADOOP_USER_NAME"] = "work"

        # print(os.environ)
        print("environ init")


def word_count():
    conf = SparkConf().setMaster('local[2]').setAppName('Word Count')
    sc = SparkContext(conf=conf)

    rdd1 = sc.parallelize([1, 2, 3, 4, 5]).map(lambda x: x + 1)
    print(rdd1.collect())

    rdd2 = sc.parallelize(["dog", "tiger", "cat", "tiger", "tiger", "cat"]).map(
        lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
    print(rdd2.collect())

    sc.stop()


def hive_sql():
    """
    1.需要将hive-site.xml放到$SPARK_HOME/conf目录下，或者创建SparkSession时指定config
    config("hive.metastore.uris", "thrift://hadoop-prod00:9083")
    2.需要在C:\\Windows\\System32\\drivers\\etc\\hosts添加集群ip和机器名的映射
    3.可选：需要将mysql-connector-java-5.1.48.jar放到$SPARK_HOME/jars目录下
    """
    spark = SparkSession.builder \
        .master("local") \
        .appName("Spark Sql") \
        .config("spark.some.config.option", "some-value") \
        .enableHiveSupport() \
        .getOrCreate()

    spark.sql("show databases").show()
    spark.sql("use tmp")
    spark.sql("show tables").show()
    df = spark.sql("select * from medical.dim_date limit 2")
    df.show()

    spark.sql("drop table tmp.dim_date_tmp")
    spark.sql("create table tmp.dim_date_tmp stored as parquet as select * from medical.dim_date limit 5")

    spark.stop()


if __name__ == '__main__':
    environ_init()
    word_count()
    hive_sql()
