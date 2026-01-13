{{
    config(
        materialized='table',
        tags=['databricks', 'neural_network']
    )
}}

/*
    MNIST Predictions - Databricks Implementation
    
    Takes the forward pass output and:
    1. Applies softmax to get probabilities
    2. Selects the class with highest probability (argmax)
*/

WITH forward_output AS (
    SELECT 
        sample_id,
        class_idx AS neuron_idx,
        logit AS activation
    FROM {{ ref('forward_pass_databricks') }}
),

probabilities AS (
    {{ databricks_softmax('forward_output') }}
),

predictions AS (
    {{ databricks_argmax('forward_output') }}
)

SELECT 
    p.sample_id,
    p.predicted_digit,
    p.confidence_score AS logit_score,
    prob.probability AS softmax_probability
FROM predictions p
LEFT JOIN probabilities prob 
    ON p.sample_id = prob.sample_id 
    AND p.predicted_digit = prob.class_idx
ORDER BY p.sample_id

