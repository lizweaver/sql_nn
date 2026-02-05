{{
    config(
        materialized='table',
        tags=['training', 'databricks', 'validation']
    )
}}

/*
    VALIDATION SOFTMAX
    
    Converts validation logits to probabilities.
*/

WITH logits AS (
    SELECT 
        sample_id,
        class_idx,
        logit
    FROM {{ ref('validation_forward_pass') }}
),

-- Numerically stable softmax
max_logits AS (
    SELECT 
        sample_id,
        MAX(logit) AS max_logit
    FROM logits
    GROUP BY sample_id
),

exp_logits AS (
    SELECT 
        l.sample_id,
        l.class_idx,
        EXP(l.logit - m.max_logit) AS exp_val
    FROM logits l
    INNER JOIN max_logits m ON l.sample_id = m.sample_id
),

sum_exp AS (
    SELECT 
        sample_id,
        SUM(exp_val) AS total_exp
    FROM exp_logits
    GROUP BY sample_id
)

SELECT 
    e.sample_id,
    e.class_idx,
    e.exp_val / s.total_exp AS probability
FROM exp_logits e
INNER JOIN sum_exp s ON e.sample_id = s.sample_id

