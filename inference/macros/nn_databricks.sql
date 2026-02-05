/* =============================================================================
   DATABRICKS NEURAL NETWORK MACROS
   
   Uses Spark SQL ARRAY functions for efficient vector operations.
   Weights are stored as arrays (one row per output neuron).
   ============================================================================= */

{% macro databricks_dot_product(vector_a, vector_b) %}
-- Computes dot product of two arrays in Databricks/Spark SQL
-- Note: Converts string representations to arrays using from_json
(
    SELECT SUM(a * b)
    FROM (
        SELECT 
            posexplode(from_json({{ vector_a }}, 'array<double>')) AS (idx_a, a)
    ) arr_a
    INNER JOIN (
        SELECT 
            posexplode(from_json({{ vector_b }}, 'array<double>')) AS (idx_b, b)
    ) arr_b
    ON arr_a.idx_a = arr_b.idx_b
)
{% endmacro %}


{% macro databricks_linear_layer(input_cte, weight_table, bias_table, apply_relu=true) %}
/*
   Performs a linear layer forward pass: output = ReLU(input @ weights.T + bias)
   
   Expects:
   - input_cte: CTE with columns (sample_id, activations as JSON string)
   - weight_table: seed/ref with columns (output_idx, weights as JSON string)
   - bias_table: seed/ref with columns (output_idx, bias_value)
*/
WITH input_exploded AS (
    SELECT 
        sample_id,
        idx as input_idx,
        val as input_val
    FROM {{ input_cte }}
    LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),
weights_exploded AS (
    SELECT 
        output_idx,
        idx as weight_idx,
        val as weight_val
    FROM {{ ref(weight_table) }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
)
SELECT 
    i.sample_id,
    w.output_idx AS neuron_idx,
    {% if apply_relu %}
    GREATEST(0, 
    {% endif %}
        SUM(i.input_val * w.weight_val) + b.bias_value
    {% if apply_relu %}
    )
    {% endif %} AS activation
FROM input_exploded i
INNER JOIN weights_exploded w ON i.input_idx = w.weight_idx
INNER JOIN {{ ref(bias_table) }} b ON w.output_idx = b.output_idx
GROUP BY i.sample_id, w.output_idx, b.bias_value
{% endmacro %}


{% macro databricks_aggregate_to_array(input_cte, value_col='activation', index_col='neuron_idx') %}
-- Aggregates neuron activations back into an array for the next layer
-- Collect as struct with index, then sort array by index, then extract values
SELECT 
    sample_id,
    to_json(
        transform(
            array_sort(collect_list(struct({{ index_col }} as idx, {{ value_col }} as val))),
            x -> x.val
        )
    ) AS activations
FROM {{ input_cte }}
GROUP BY sample_id
{% endmacro %}


{% macro databricks_pixels_to_array() %}
-- Converts pixel-format input (sample_id, pixel_idx, pixel_value) to array format
-- Collect as struct with index, then sort array by index, then extract values
SELECT 
    sample_id,
    to_json(
        transform(
            array_sort(collect_list(struct(pixel_idx as idx, pixel_value as val))),
            x -> x.val
        )
    ) AS activations
FROM {{ ref('input_images') }}
GROUP BY sample_id
{% endmacro %}


{% macro databricks_softmax(input_cte) %}
-- Applies softmax to convert logits to probabilities
WITH exp_logits AS (
    SELECT 
        sample_id,
        neuron_idx AS class_idx,
        EXP(activation) AS exp_val
    FROM {{ input_cte }}
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
{% endmacro %}


{% macro databricks_argmax(input_cte) %}
-- Returns the class with highest activation/probability
WITH ranked AS (
    SELECT 
        sample_id,
        neuron_idx AS class_idx,
        activation,
        ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY activation DESC) AS rank
    FROM {{ input_cte }}
)
SELECT 
    sample_id,
    class_idx AS predicted_digit,
    activation AS confidence_score
FROM ranked
WHERE rank = 1
{% endmacro %}

