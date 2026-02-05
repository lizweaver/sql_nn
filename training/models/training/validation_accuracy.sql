{{
    config(
        materialized='table',
        tags=['training', 'databricks', 'validation']
    )
}}

/*
    VALIDATION ACCURACY
    
    Computes accuracy on the validation set.
*/

WITH predictions AS (
    SELECT 
        sample_id,
        class_idx AS predicted_digit,
        probability AS confidence,
        ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY probability DESC) AS rank
    FROM {{ ref('validation_softmax') }}
),

top_predictions AS (
    SELECT 
        sample_id,
        predicted_digit,
        confidence
    FROM predictions
    WHERE rank = 1
),

results AS (
    SELECT 
        p.sample_id,
        p.predicted_digit,
        t.true_label,
        CASE WHEN p.predicted_digit = t.true_label THEN 1 ELSE 0 END AS correct
    FROM top_predictions p
    INNER JOIN {{ ref('validation_labels') }} t ON p.sample_id = t.sample_id
),

loss_info AS (
    SELECT average_loss FROM {{ ref('validation_loss') }}
)

SELECT 
    SUM(correct) AS correct_count,
    COUNT(*) AS total_count,
    CAST(SUM(correct) AS DOUBLE) / COUNT(*) AS accuracy,
    l.average_loss AS loss
FROM results
CROSS JOIN loss_info l
GROUP BY l.average_loss

