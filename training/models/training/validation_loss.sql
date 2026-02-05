{{
    config(
        materialized='table',
        tags=['training', 'databricks', 'validation']
    )
}}

/*
    VALIDATION LOSS
    
    Computes cross-entropy loss on the validation set.
    This is used to detect overfitting (when training loss decreases
    but validation loss increases).
*/

WITH correct_probs AS (
    SELECT 
        p.sample_id,
        p.class_idx,
        p.probability,
        t.true_label
    FROM {{ ref('validation_softmax') }} p
    INNER JOIN {{ ref('validation_labels') }} t 
        ON p.sample_id = t.sample_id AND p.class_idx = t.true_label
),

sample_losses AS (
    SELECT 
        sample_id,
        true_label,
        probability,
        -LN(GREATEST(probability, 1e-10)) AS loss
    FROM correct_probs
)

SELECT 
    AVG(loss) AS average_loss,
    SUM(loss) AS total_loss,
    COUNT(*) AS batch_size,
    MIN(loss) AS min_loss,
    MAX(loss) AS max_loss
FROM sample_losses

