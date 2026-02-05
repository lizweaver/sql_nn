{{
    config(
        materialized='table',
        tags=['training', 'databricks', 'sgd', 'optimized']
    )
}}

/*
    OPTIMIZED MINI-BATCH SGD TRAINING STEP
    
    Combines forward pass, backward pass, and weight updates into a SINGLE query,
    but only processes a mini-batch of samples for stochastic gradient descent.
    
    Usage:
    dbt run --select training_step_sgd --vars '{"batch_id": 0, "batch_size": 32, "learning_rate": 0.01}'
    
    Benefits of SGD:
    - Noisier gradients help escape local minima
    - Multiple weight updates per epoch
    - Can work with larger datasets
*/

{% set batch_id = var('batch_id', 0) %}
{% set batch_size = var('batch_size', 32) %}
{% set learning_rate = var('learning_rate', 0.01) %}

-- ============================================================================
-- BATCH SELECTION
-- ============================================================================
WITH all_samples AS (
    SELECT DISTINCT sample_id FROM {{ ref('input_images') }}
),
sample_batches AS (
    SELECT sample_id,
           CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY sample_id) - 1) / {{ batch_size }}) AS INT) AS batch_num
    FROM all_samples
),
current_batch AS (
    SELECT sample_id FROM sample_batches WHERE batch_num = {{ batch_id }}
),

-- ============================================================================
-- FORWARD PASS (mini-batch only)
-- ============================================================================
input_arrays AS (
    SELECT i.sample_id,
           to_json(transform(array_sort(collect_list(struct(i.pixel_idx AS idx, i.pixel_value AS val))), x -> x.val)) AS activations
    FROM {{ ref('input_images') }} i
    INNER JOIN current_batch b ON i.sample_id = b.sample_id
    GROUP BY i.sample_id
),

-- Layer 1: Linear(784, 128) + ReLU
l1_input AS (
    SELECT sample_id, idx AS input_idx, val AS input_val
    FROM input_arrays LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),
l1_weights AS (
    SELECT output_idx, idx AS weight_idx, val AS weight_val
    FROM {{ ref('weights_fc1') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
l1_pre AS (
    SELECT i.sample_id, w.output_idx AS neuron_idx,
           SUM(i.input_val * w.weight_val) + b.bias_value AS z
    FROM l1_input i
    JOIN l1_weights w ON i.input_idx = w.weight_idx
    JOIN {{ ref('biases_fc1') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
l1_act AS (
    SELECT sample_id, neuron_idx, GREATEST(0.0, z) AS a, z FROM l1_pre
),
l1_arr AS (
    SELECT sample_id, to_json(transform(array_sort(collect_list(struct(neuron_idx AS idx, a AS val))), x -> x.val)) AS activations
    FROM l1_act GROUP BY sample_id
),

-- Layer 2: Linear(128, 64) + ReLU
l2_input AS (
    SELECT sample_id, idx AS input_idx, val AS input_val
    FROM l1_arr LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),
l2_weights AS (
    SELECT output_idx, idx AS weight_idx, val AS weight_val
    FROM {{ ref('weights_fc2') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
l2_pre AS (
    SELECT i.sample_id, w.output_idx AS neuron_idx,
           SUM(i.input_val * w.weight_val) + b.bias_value AS z
    FROM l2_input i
    JOIN l2_weights w ON i.input_idx = w.weight_idx
    JOIN {{ ref('biases_fc2') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
l2_act AS (
    SELECT sample_id, neuron_idx, GREATEST(0.0, z) AS a, z FROM l2_pre
),
l2_arr AS (
    SELECT sample_id, to_json(transform(array_sort(collect_list(struct(neuron_idx AS idx, a AS val))), x -> x.val)) AS activations
    FROM l2_act GROUP BY sample_id
),

-- Layer 3: Linear(64, 10) - output logits
l3_input AS (
    SELECT sample_id, idx AS input_idx, val AS input_val
    FROM l2_arr LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),
l3_weights AS (
    SELECT output_idx, idx AS weight_idx, val AS weight_val
    FROM {{ ref('weights_fc3') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
l3_logits AS (
    SELECT i.sample_id, w.output_idx AS neuron_idx,
           SUM(i.input_val * w.weight_val) + b.bias_value AS logit
    FROM l3_input i
    JOIN l3_weights w ON i.input_idx = w.weight_idx
    JOIN {{ ref('biases_fc3') }} b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),

-- Softmax (numerically stable)
max_logits AS (SELECT sample_id, MAX(logit) AS max_l FROM l3_logits GROUP BY sample_id),
exp_logits AS (
    SELECT l.sample_id, l.neuron_idx, EXP(l.logit - m.max_l) AS exp_val
    FROM l3_logits l JOIN max_logits m ON l.sample_id = m.sample_id
),
sum_exp AS (SELECT sample_id, SUM(exp_val) AS total FROM exp_logits GROUP BY sample_id),
softmax AS (
    SELECT e.sample_id, e.neuron_idx AS class_idx, e.exp_val / s.total AS prob
    FROM exp_logits e JOIN sum_exp s ON e.sample_id = s.sample_id
),

-- ============================================================================
-- BACKWARD PASS (gradients from mini-batch)
-- ============================================================================
batch_labels AS (
    SELECT t.sample_id, t.true_label
    FROM {{ ref('true_labels') }} t
    INNER JOIN current_batch b ON t.sample_id = b.sample_id
),
n_samples AS (SELECT COUNT(*) AS n FROM current_batch),

-- Output gradient: dL/dz3 = softmax - one_hot(y)
dz3 AS (
    SELECT p.sample_id, p.class_idx AS neuron_idx,
           p.prob - CASE WHEN p.class_idx = t.true_label THEN 1.0 ELSE 0.0 END AS grad
    FROM softmax p JOIN batch_labels t ON p.sample_id = t.sample_id
),

-- Layer 3 gradients
grad_w3 AS (
    SELECT g.neuron_idx AS out_idx, a.neuron_idx AS in_idx,
           SUM(g.grad * a.a) / ns.n AS dw
    FROM dz3 g JOIN l2_act a ON g.sample_id = a.sample_id CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, a.neuron_idx, ns.n
),
grad_b3 AS (
    SELECT neuron_idx AS out_idx, SUM(grad) / ns.n AS db
    FROM dz3 CROSS JOIN n_samples ns GROUP BY neuron_idx, ns.n
),

-- Backprop through layer 3
da2 AS (
    SELECT g.sample_id, w.weight_idx AS neuron_idx, SUM(g.grad * w.weight_val) AS grad
    FROM dz3 g JOIN l3_weights w ON g.neuron_idx = w.output_idx
    GROUP BY g.sample_id, w.weight_idx
),
dz2 AS (
    SELECT g.sample_id, g.neuron_idx,
           CASE WHEN a.z > 0 THEN g.grad ELSE 0.0 END AS grad
    FROM da2 g JOIN l2_act a ON g.sample_id = a.sample_id AND g.neuron_idx = a.neuron_idx
),

-- Layer 2 gradients
grad_w2 AS (
    SELECT g.neuron_idx AS out_idx, a.neuron_idx AS in_idx,
           SUM(g.grad * a.a) / ns.n AS dw
    FROM dz2 g JOIN l1_act a ON g.sample_id = a.sample_id CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, a.neuron_idx, ns.n
),
grad_b2 AS (
    SELECT neuron_idx AS out_idx, SUM(grad) / ns.n AS db
    FROM dz2 CROSS JOIN n_samples ns GROUP BY neuron_idx, ns.n
),

-- Backprop through layer 2
da1 AS (
    SELECT g.sample_id, w.weight_idx AS neuron_idx, SUM(g.grad * w.weight_val) AS grad
    FROM dz2 g JOIN l2_weights w ON g.neuron_idx = w.output_idx
    GROUP BY g.sample_id, w.weight_idx
),
dz1 AS (
    SELECT g.sample_id, g.neuron_idx,
           CASE WHEN a.z > 0 THEN g.grad ELSE 0.0 END AS grad
    FROM da1 g JOIN l1_act a ON g.sample_id = a.sample_id AND g.neuron_idx = a.neuron_idx
),

-- Layer 1 gradients
l0_act AS (
    SELECT sample_id, idx AS neuron_idx, val AS a
    FROM input_arrays LATERAL VIEW posexplode(from_json(activations, 'array<double>')) AS idx, val
),
grad_w1 AS (
    SELECT g.neuron_idx AS out_idx, a.neuron_idx AS in_idx,
           SUM(g.grad * a.a) / ns.n AS dw
    FROM dz1 g JOIN l0_act a ON g.sample_id = a.sample_id CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, a.neuron_idx, ns.n
),
grad_b1 AS (
    SELECT neuron_idx AS out_idx, SUM(grad) / ns.n AS db
    FROM dz1 CROSS JOIN n_samples ns GROUP BY neuron_idx, ns.n
),

-- ============================================================================
-- WEIGHT UPDATES: W_new = W_old - lr * grad
-- ============================================================================
fc1_old AS (
    SELECT output_idx, idx AS input_idx, val AS w
    FROM {{ ref('weights_fc1') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
fc1_new AS (
    SELECT w.output_idx, w.input_idx, w.w - {{ learning_rate }} * COALESCE(g.dw, 0.0) AS new_w
    FROM fc1_old w LEFT JOIN grad_w1 g ON w.output_idx = g.out_idx AND w.input_idx = g.in_idx
),
fc1_agg AS (
    SELECT output_idx, to_json(transform(array_sort(collect_list(struct(input_idx AS idx, new_w AS val))), x -> x.val)) AS weights
    FROM fc1_new GROUP BY output_idx
),
fc1_bias_new AS (
    SELECT b.output_idx, b.bias_value - {{ learning_rate }} * COALESCE(g.db, 0.0) AS bias_value
    FROM {{ ref('biases_fc1') }} b LEFT JOIN grad_b1 g ON b.output_idx = g.out_idx
),

fc2_old AS (
    SELECT output_idx, idx AS input_idx, val AS w
    FROM {{ ref('weights_fc2') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
fc2_new AS (
    SELECT w.output_idx, w.input_idx, w.w - {{ learning_rate }} * COALESCE(g.dw, 0.0) AS new_w
    FROM fc2_old w LEFT JOIN grad_w2 g ON w.output_idx = g.out_idx AND w.input_idx = g.in_idx
),
fc2_agg AS (
    SELECT output_idx, to_json(transform(array_sort(collect_list(struct(input_idx AS idx, new_w AS val))), x -> x.val)) AS weights
    FROM fc2_new GROUP BY output_idx
),
fc2_bias_new AS (
    SELECT b.output_idx, b.bias_value - {{ learning_rate }} * COALESCE(g.db, 0.0) AS bias_value
    FROM {{ ref('biases_fc2') }} b LEFT JOIN grad_b2 g ON b.output_idx = g.out_idx
),

fc3_old AS (
    SELECT output_idx, idx AS input_idx, val AS w
    FROM {{ ref('weights_fc3') }} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
fc3_new AS (
    SELECT w.output_idx, w.input_idx, w.w - {{ learning_rate }} * COALESCE(g.dw, 0.0) AS new_w
    FROM fc3_old w LEFT JOIN grad_w3 g ON w.output_idx = g.out_idx AND w.input_idx = g.in_idx
),
fc3_agg AS (
    SELECT output_idx, to_json(transform(array_sort(collect_list(struct(input_idx AS idx, new_w AS val))), x -> x.val)) AS weights
    FROM fc3_new GROUP BY output_idx
),
fc3_bias_new AS (
    SELECT b.output_idx, b.bias_value - {{ learning_rate }} * COALESCE(g.db, 0.0) AS bias_value
    FROM {{ ref('biases_fc3') }} b LEFT JOIN grad_b3 g ON b.output_idx = g.out_idx
),

-- ============================================================================
-- BATCH METRICS
-- ============================================================================
batch_loss AS (
    SELECT -AVG(LN(GREATEST(p.prob, 1e-10))) AS loss
    FROM softmax p JOIN batch_labels t 
    ON p.sample_id = t.sample_id AND p.class_idx = t.true_label
),
batch_preds AS (
    SELECT sample_id, class_idx, ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY prob DESC) AS rn
    FROM softmax
),
batch_acc AS (
    SELECT CAST(SUM(CASE WHEN p.class_idx = t.true_label THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS accuracy
    FROM batch_preds p JOIN batch_labels t ON p.sample_id = t.sample_id WHERE p.rn = 1
)

-- ============================================================================
-- OUTPUT
-- ============================================================================
SELECT 'weights_fc1' AS param, output_idx, NULL AS bias_value, weights, NULL AS loss, NULL AS accuracy FROM fc1_agg
UNION ALL SELECT 'biases_fc1', output_idx, bias_value, NULL, NULL, NULL FROM fc1_bias_new
UNION ALL SELECT 'weights_fc2', output_idx, NULL, weights, NULL, NULL FROM fc2_agg
UNION ALL SELECT 'biases_fc2', output_idx, bias_value, NULL, NULL, NULL FROM fc2_bias_new
UNION ALL SELECT 'weights_fc3', output_idx, NULL, weights, NULL, NULL FROM fc3_agg
UNION ALL SELECT 'biases_fc3', output_idx, bias_value, NULL, NULL, NULL FROM fc3_bias_new
UNION ALL SELECT 'metrics', 0, NULL, NULL, loss, accuracy FROM batch_loss CROSS JOIN batch_acc

