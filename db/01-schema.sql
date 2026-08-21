-- Freight rate prediction: prediction and outcome logging.
--
-- Two tables kept deliberately separate. A rate is quoted now and confirmed
-- days or weeks later, so predictions and actuals do not arrive together and
-- cannot be written in one row. Joining them on load_id is what lets us score
-- the model after the fact.

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       UUID PRIMARY KEY,

    -- Supplied by the caller so an actual can be matched back later. Null when
    -- the caller did not provide one, in which case the row is still useful for
    -- drift monitoring but can never be scored.
    load_id             TEXT,

    -- The request as received, kept so drift can be measured on real traffic
    -- rather than on whatever the model happened to derive from it.
    pickup              TEXT        NOT NULL,
    delivery            TEXT        NOT NULL,
    distance            REAL        NOT NULL,
    equipment           TEXT        NOT NULL,
    weight              REAL,
    load_date           DATE        NOT NULL,

    predicted_rate      REAL        NOT NULL,
    rate_per_mile       REAL        NOT NULL,
    model_version       TEXT        NOT NULL,

    -- Flags raised at prediction time. unknown_city is the most useful single
    -- drift signal we have: it rises when traffic moves into markets the model
    -- has no history for.
    unknown_city            BOOLEAN NOT NULL DEFAULT FALSE,
    date_beyond_training    BOOLEAN NOT NULL DEFAULT FALSE,
    distance_out_of_range   BOOLEAN NOT NULL DEFAULT FALSE,
    weight_imputed          BOOLEAN NOT NULL DEFAULT FALSE,

    latency_ms          REAL,
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Almost every query filters by time, so this index carries the dashboard.
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at
    ON predictions (predicted_at DESC);

-- Used by the join against actuals.
CREATE INDEX IF NOT EXISTS idx_predictions_load_id
    ON predictions (load_id)
    WHERE load_id IS NOT NULL;

-- Lets performance be broken down by model version after a retrain.
CREATE INDEX IF NOT EXISTS idx_predictions_model_version
    ON predictions (model_version);


CREATE TABLE IF NOT EXISTS actuals (
    load_id             TEXT PRIMARY KEY,
    actual_rate         REAL        NOT NULL CHECK (actual_rate > 0),

    -- Where the number came from, so a manual correction can be told apart
    -- from a feed.
    source              TEXT        NOT NULL DEFAULT 'api',
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_actuals_recorded_at
    ON actuals (recorded_at DESC);


-- Predictions that can be scored, with the error already computed. Reading
-- from a view keeps the error arithmetic in one place rather than repeated in
-- every query that needs it.
CREATE OR REPLACE VIEW scored_predictions AS
SELECT
    p.prediction_id,
    p.load_id,
    p.pickup,
    p.delivery,
    p.distance,
    p.equipment,
    p.load_date,
    p.predicted_rate,
    p.model_version,
    p.unknown_city,
    p.predicted_at,
    a.actual_rate,
    a.recorded_at,
    a.actual_rate - p.predicted_rate                        AS error,
    ABS(a.actual_rate - p.predicted_rate)                   AS absolute_error,
    ABS(a.actual_rate - p.predicted_rate) / a.actual_rate   AS absolute_pct_error,

    -- How long the outcome took to arrive. Worth watching on its own: if this
    -- grows, the model is being scored on ever staler feedback.
    EXTRACT(EPOCH FROM (a.recorded_at - p.predicted_at)) / 86400.0 AS feedback_days
FROM predictions p
INNER JOIN actuals a ON a.load_id = p.load_id;