-- Runs before the monitoring schema, because MLflow needs its own database and
-- the Postgres image only creates the one named in POSTGRES_DB.
--
-- Keeping tracking data separate from monitoring data means a schema change to
-- one cannot disturb the other, and either can be dropped without the other
-- noticing.

CREATE DATABASE mlflow;

GRANT ALL PRIVILEGES ON DATABASE mlflow TO freight;