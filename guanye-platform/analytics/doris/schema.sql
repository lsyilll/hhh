CREATE DATABASE IF NOT EXISTS guanye;
CREATE TABLE IF NOT EXISTS guanye.ods_events(event_id VARCHAR(64),event_type VARCHAR(80),project_id VARCHAR(64),owner_id VARCHAR(64),task_id VARCHAR(64),status VARCHAR(20),duration_ms BIGINT,ts_ms BIGINT,payload STRING)
UNIQUE KEY(event_id) DISTRIBUTED BY HASH(event_id) BUCKETS 4 PROPERTIES('replication_num'='1','enable_unique_key_merge_on_write'='true');
CREATE VIEW IF NOT EXISTS guanye.dwd_tasks AS SELECT event_id,project_id,task_id,status,duration_ms,FROM_UNIXTIME(ts_ms/1000) AS occurred_at FROM guanye.ods_events WHERE event_type IN ('task.complete','task.failed');
CREATE TABLE IF NOT EXISTS guanye.dws_project_minute(project_id VARCHAR(64),window_start DATETIME,window_end DATETIME,completed BIGINT,failed BIGINT,avg_duration DOUBLE,owner_id VARCHAR(64))
UNIQUE KEY(project_id,window_start) DISTRIBUTED BY HASH(project_id) BUCKETS 4 PROPERTIES('replication_num'='1');
CREATE VIEW IF NOT EXISTS guanye.ads_daily_quality AS SELECT project_id,DATE(occurred_at) AS day,COUNT(*) AS tasks,SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failures,AVG(duration_ms) AS average_ms FROM guanye.dwd_tasks GROUP BY project_id,DATE(occurred_at);
