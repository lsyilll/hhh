SET 'execution.attached' = 'false';
SET 'table.dml-sync' = 'false';
SET 'pipeline.name' = 'guanye-minute-metrics';
SET 'execution.checkpointing.interval' = '30 s';
SET 'table.local-time-zone' = 'Asia/Shanghai';
CREATE TABLE events (
 event_id STRING,event_type STRING,owner_id STRING,project_id STRING,task_id STRING,status STRING,duration_ms BIGINT,ts_ms BIGINT,
 event_time AS TO_TIMESTAMP_LTZ(ts_ms,3), WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND
) WITH ('connector'='kafka','topic'='guanye.events','properties.bootstrap.servers'='kafka:9092','properties.group.id'='guanye-flink-metrics','scan.startup.mode'='earliest-offset','format'='json','json.ignore-parse-errors'='true');
CREATE VIEW deduplicated AS SELECT event_id,event_type,owner_id,project_id,task_id,status,duration_ms,event_time FROM (
 SELECT *,ROW_NUMBER() OVER(PARTITION BY event_id ORDER BY event_time ASC) AS row_num FROM events
) WHERE row_num=1;
CREATE TABLE minute_metrics (owner_id STRING,project_id STRING,window_start TIMESTAMP(3),window_end TIMESTAMP(3),completed BIGINT,failed BIGINT,avg_duration DOUBLE,PRIMARY KEY(project_id,window_start) NOT ENFORCED)
WITH ('connector'='upsert-kafka','topic'='guanye.metrics.minute','properties.bootstrap.servers'='kafka:9092','key.format'='json','value.format'='json');
INSERT INTO minute_metrics SELECT owner_id,project_id,window_start,window_end,
 SUM(CASE WHEN event_type='task.complete' THEN CAST(1 AS BIGINT) ELSE CAST(0 AS BIGINT) END),
 SUM(CASE WHEN event_type='task.failed' THEN CAST(1 AS BIGINT) ELSE CAST(0 AS BIGINT) END),AVG(CAST(duration_ms AS DOUBLE))
 FROM TABLE(TUMBLE(TABLE deduplicated,DESCRIPTOR(event_time),INTERVAL '1' MINUTE)) GROUP BY owner_id,project_id,window_start,window_end;
