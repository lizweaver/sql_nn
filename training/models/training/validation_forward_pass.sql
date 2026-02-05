{{
    config(
        materialized='table',
        tags=['training', 'databricks', 'validation']
    )
}}

/*
    VALIDATION FORWARD PASS
    
    Runs the forward pass on validation data (separate from training data).
    Uses the same weights but different input samples.
*/

-- Step 1: Convert validation input pixels to array format
WITH input_arrays AS (
    SELECT 
        sample_id,
        to_json(
            transform(
                array_sort(collect_list(struct(pixel_idx AS idx, pixel_value AS val))),
                x -> x.val
            )
        ) AS activations
    FROM {{ ref('validation_images') }}
    GROUP BY sample_id
),

-- ============================================================================
-- LAYER 1: Linear(784, 128) + ReLU
-- ============================================================================
layer1_input_exploded AS (
    SELECT 
        sample_id,
        idx AS input_idx,
        val AS input_val
    FROM input_arrays
    LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),

layer1_weights_exploded AS (
    SELECT 
        output_idx,
        idx AS weight_idx,
        val AS weight_val
    FROM {{ ref('weights_fc1') }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),

layer1_activation AS (
    SELECT 
        i.sample_id,
        w.output_idx AS neuron_idx,
        GREATEST(0.0, SUM(i.input_val * w.weight_val) + b.bias_value) AS activation
    FROM layer1_input_exploded i
    INNER JOIN layer1_weights_exploded w ON i.input_idx = w.weight_idx
    INNER JOIN {{ ref('biases_fc1') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),

layer1_arrays AS (
    SELECT 
        sample_id,
        to_json(
            transform(
                array_sort(collect_list(struct(neuron_idx AS idx, activation AS val))),
                x -> x.val
            )
        ) AS activations
    FROM layer1_activation
    GROUP BY sample_id
),

-- ============================================================================
-- LAYER 2: Linear(128, 64) + ReLU
-- ============================================================================
layer2_input_exploded AS (
    SELECT 
        sample_id,
        idx AS input_idx,
        val AS input_val
    FROM layer1_arrays
    LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),

layer2_weights_exploded AS (
    SELECT 
        output_idx,
        idx AS weight_idx,
        val AS weight_val
    FROM {{ ref('weights_fc2') }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),

layer2_activation AS (
    SELECT 
        i.sample_id,
        w.output_idx AS neuron_idx,
        GREATEST(0.0, SUM(i.input_val * w.weight_val) + b.bias_value) AS activation
    FROM layer2_input_exploded i
    INNER JOIN layer2_weights_exploded w ON i.input_idx = w.weight_idx
    INNER JOIN {{ ref('biases_fc2') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),

layer2_arrays AS (
    SELECT 
        sample_id,
        to_json(
            transform(
                array_sort(collect_list(struct(neuron_idx AS idx, activation AS val))),
                x -> x.val
            )
        ) AS activations
    FROM layer2_activation
    GROUP BY sample_id
),

-- ============================================================================
-- LAYER 3: Linear(64, 10) - Output logits
-- ============================================================================
layer3_input_exploded AS (
    SELECT 
        sample_id,
        idx AS input_idx,
        val AS input_val
    FROM layer2_arrays
    LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),

layer3_weights_exploded AS (
    SELECT 
        output_idx,
        idx AS weight_idx,
        val AS weight_val
    FROM {{ ref('weights_fc3') }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),

layer3_logits AS (
    SELECT 
        i.sample_id,
        w.output_idx AS neuron_idx,
        SUM(i.input_val * w.weight_val) + b.bias_value AS logit
    FROM layer3_input_exploded i
    INNER JOIN layer3_weights_exploded w ON i.input_idx = w.weight_idx
    INNER JOIN {{ ref('biases_fc3') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
)

-- Output logits for validation samples
SELECT 
    sample_id,
    neuron_idx AS class_idx,
    logit
FROM layer3_logits

