FROM flink:1.20.3-scala_2.12-java17
ADD --checksum=sha256:b860724fd97d7da0901bfc3859680442f4c1aa2b35ab944407558b393a1cf93e https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.4.0-1.20/flink-sql-connector-kafka-3.4.0-1.20.jar /opt/flink/lib/kafka.jar
RUN chmod 644 /opt/flink/lib/kafka.jar
