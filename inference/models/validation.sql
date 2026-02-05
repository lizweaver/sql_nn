{{
    config(
        materialized='table',
        tags=['validation']
    )
}}

/*
    Validation Model
    
    Compares predictions from Databricks implementation against ground truth.
    Use this to verify that the SQL implementation matches the PyTorch model.
*/

WITH databricks_predictions AS (
    SELECT 
        sample_id,
        predicted_digit AS databricks_prediction,
        logit_score AS databricks_logit
    FROM {{ ref('predictions_databricks') }}
),

ground_truth AS (
    SELECT 
        sample_id,
        true_label
    FROM {{ ref('true_labels') }}
)

SELECT 
    gt.sample_id,
    gt.true_label,
    d.databricks_prediction,
    d.databricks_logit,
    CASE WHEN d.databricks_prediction = gt.true_label THEN 1 ELSE 0 END AS databricks_correct
FROM ground_truth gt
LEFT JOIN databricks_predictions d ON gt.sample_id = d.sample_id
ORDER BY gt.sample_id
