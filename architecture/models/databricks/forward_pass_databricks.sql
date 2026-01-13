{{
    config(
        materialized='table',
        tags=['databricks', 'neural_network']
    )
}}

/*
    MNIST Neural Network Forward Pass - Databricks Implementation
    
    Architecture:
    - Input: 784 pixels (28x28 flattened)
    - fc1: Linear(784, 128) + ReLU
    - fc2: Linear(128, 64) + ReLU
    - fc3: Linear(64, 10) - output logits
    
    This implementation uses Spark SQL ARRAY functions for efficient
    vector operations. Weights are stored as arrays (one row per output neuron).
*/

-- Step 1: Convert input pixels to array format
WITH input_arrays AS (
    {{ databricks_pixels_to_array() }}
),

-- Step 2: Layer 1 - Linear(784, 128) + ReLU
layer1_neurons AS (
    {{ databricks_linear_layer('input_arrays', 'fc1_weights_databricks', 'fc1_bias', apply_relu=true) }}
),

-- Aggregate layer 1 outputs back to array for next layer
layer1_arrays AS (
    {{ databricks_aggregate_to_array('layer1_neurons') }}
),

-- Step 3: Layer 2 - Linear(128, 64) + ReLU
layer2_neurons AS (
    {{ databricks_linear_layer('layer1_arrays', 'fc2_weights_databricks', 'fc2_bias', apply_relu=true) }}
),

-- Aggregate layer 2 outputs back to array for next layer
layer2_arrays AS (
    {{ databricks_aggregate_to_array('layer2_neurons') }}
),

-- Step 4: Layer 3 - Linear(64, 10) - No activation (raw logits)
output_logits AS (
    {{ databricks_linear_layer('layer2_arrays', 'fc3_weights_databricks', 'fc3_bias', apply_relu=false) }}
)

SELECT 
    sample_id,
    neuron_idx AS class_idx,
    activation AS logit
FROM output_logits

