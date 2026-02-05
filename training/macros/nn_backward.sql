/* =============================================================================
   SQL NEURAL NETWORK - BACKWARD PASS MACROS
   
   Implements backpropagation for training a neural network in SQL.
   Uses chain rule to compute gradients through the network.
   
   Architecture: 784 → 128 → 64 → 10 (same as inference)
   ============================================================================= */


/* -----------------------------------------------------------------------------
   SOFTMAX CROSS-ENTROPY GRADIENT
   
   For softmax + cross-entropy, the gradient simplifies to:
   dL/dz = softmax(z) - y_onehot = p - y
   
   This is the starting point for backpropagation.
   ----------------------------------------------------------------------------- */
{% macro backward_output_gradient() %}
-- Computes gradient of loss with respect to output logits
-- dz3 = softmax(logits) - one_hot(true_label)
-- Shape: (batch_size, 10)
WITH softmax_probs AS (
    SELECT 
        sample_id,
        class_idx,
        probability
    FROM {{ ref('forward_pass_softmax') }}
),
one_hot_labels AS (
    SELECT 
        sample_id,
        class_idx,
        CASE WHEN class_idx = true_label THEN 1.0 ELSE 0.0 END AS target
    FROM {{ ref('true_labels') }} t
    CROSS JOIN (SELECT DISTINCT class_idx FROM softmax_probs) c
)
SELECT 
    p.sample_id,
    p.class_idx AS neuron_idx,
    p.probability - o.target AS gradient
FROM softmax_probs p
INNER JOIN one_hot_labels o 
    ON p.sample_id = o.sample_id AND p.class_idx = o.class_idx
{% endmacro %}


/* -----------------------------------------------------------------------------
   WEIGHT GRADIENT COMPUTATION
   
   For a linear layer y = x @ W.T + b:
   dL/dW = dL/dy.T @ x  (outer product of gradient and input)
   dL/db = sum(dL/dy)   (sum over batch)
   
   We compute gradients averaged over the batch for stable training.
   ----------------------------------------------------------------------------- */
{% macro backward_weight_gradient(gradient_cte, activation_cte, weight_table) %}
-- Computes dL/dW for a linear layer
-- gradient_cte: CTE with (sample_id, neuron_idx, gradient) - shape (batch, out_dim)
-- activation_cte: CTE with (sample_id, neuron_idx, activation) - shape (batch, in_dim)
-- Result: (output_idx, input_idx, weight_gradient)
WITH grad_expanded AS (
    SELECT 
        sample_id,
        neuron_idx AS output_idx,
        gradient
    FROM {{ gradient_cte }}
),
act_expanded AS (
    SELECT 
        sample_id,
        neuron_idx AS input_idx,
        activation
    FROM {{ activation_cte }}
),
batch_count AS (
    SELECT COUNT(DISTINCT sample_id) AS n_samples
    FROM grad_expanded
)
-- dW[i,j] = mean over batch of (dL/dy[i] * x[j])
SELECT 
    g.output_idx,
    a.input_idx,
    SUM(g.gradient * a.activation) / bc.n_samples AS weight_gradient
FROM grad_expanded g
INNER JOIN act_expanded a ON g.sample_id = a.sample_id
CROSS JOIN batch_count bc
GROUP BY g.output_idx, a.input_idx, bc.n_samples
{% endmacro %}


{% macro backward_bias_gradient(gradient_cte) %}
-- Computes dL/db for a linear layer
-- gradient_cte: CTE with (sample_id, neuron_idx, gradient)
-- Result: (output_idx, bias_gradient)
WITH batch_count AS (
    SELECT COUNT(DISTINCT sample_id) AS n_samples
    FROM {{ gradient_cte }}
)
SELECT 
    neuron_idx AS output_idx,
    SUM(gradient) / bc.n_samples AS bias_gradient
FROM {{ gradient_cte }}
CROSS JOIN batch_count bc
GROUP BY neuron_idx, bc.n_samples
{% endmacro %}


/* -----------------------------------------------------------------------------
   BACKPROPAGATE THROUGH LINEAR LAYER
   
   dL/dx = dL/dy @ W  (matrix multiply with weights)
   
   This propagates the gradient backward to the previous layer.
   ----------------------------------------------------------------------------- */
{% macro backward_through_linear(gradient_cte, weight_table) %}
-- Backpropagates gradient through a linear layer
-- gradient_cte: CTE with (sample_id, neuron_idx, gradient) - gradient w.r.t output
-- weight_table: Table with (output_idx, weights as JSON array)
-- Result: (sample_id, neuron_idx, gradient) - gradient w.r.t input
WITH grad_expanded AS (
    SELECT 
        sample_id,
        neuron_idx AS output_idx,
        gradient
    FROM {{ gradient_cte }}
),
weights_exploded AS (
    SELECT 
        output_idx,
        idx AS input_idx,
        val AS weight_val
    FROM {{ ref(weight_table) }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
)
-- dx[j] = sum over i of (dL/dy[i] * W[i,j])
SELECT 
    g.sample_id,
    w.input_idx AS neuron_idx,
    SUM(g.gradient * w.weight_val) AS gradient
FROM grad_expanded g
INNER JOIN weights_exploded w ON g.output_idx = w.output_idx
GROUP BY g.sample_id, w.input_idx
{% endmacro %}


/* -----------------------------------------------------------------------------
   RELU BACKWARD PASS
   
   ReLU derivative: f'(x) = 1 if x > 0, else 0
   
   We need the PRE-activation values (before ReLU) to compute this.
   dL/dz = dL/da * (z > 0 ? 1 : 0)
   ----------------------------------------------------------------------------- */
{% macro backward_relu(gradient_cte, preactivation_cte) %}
-- Backpropagates gradient through ReLU activation
-- gradient_cte: CTE with (sample_id, neuron_idx, gradient) - gradient w.r.t output of ReLU
-- preactivation_cte: CTE with (sample_id, neuron_idx, preactivation) - values BEFORE ReLU
-- Result: (sample_id, neuron_idx, gradient) - gradient w.r.t input of ReLU
SELECT 
    g.sample_id,
    g.neuron_idx,
    CASE 
        WHEN p.preactivation > 0 THEN g.gradient 
        ELSE 0.0 
    END AS gradient
FROM {{ gradient_cte }} g
INNER JOIN {{ preactivation_cte }} p 
    ON g.sample_id = p.sample_id AND g.neuron_idx = p.neuron_idx
{% endmacro %}


/* -----------------------------------------------------------------------------
   WEIGHT UPDATE (GRADIENT DESCENT)
   
   W_new = W_old - learning_rate * dL/dW
   b_new = b_old - learning_rate * dL/db
   ----------------------------------------------------------------------------- */
{% macro update_weights(weight_table, gradient_table, learning_rate=0.01) %}
-- Updates weights using gradient descent
-- Produces new weight table with updated values
WITH old_weights AS (
    SELECT 
        output_idx,
        idx AS input_idx,
        val AS old_weight
    FROM {{ ref(weight_table) }}
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
gradients AS (
    SELECT 
        output_idx,
        input_idx,
        weight_gradient
    FROM {{ gradient_table }}
),
updated_weights AS (
    SELECT 
        w.output_idx,
        w.input_idx,
        w.old_weight - {{ learning_rate }} * COALESCE(g.weight_gradient, 0.0) AS new_weight
    FROM old_weights w
    LEFT JOIN gradients g 
        ON w.output_idx = g.output_idx AND w.input_idx = g.input_idx
)
-- Reaggregate into array format
SELECT 
    output_idx,
    to_json(
        transform(
            array_sort(collect_list(struct(input_idx AS idx, new_weight AS val))),
            x -> x.val
        )
    ) AS weights
FROM updated_weights
GROUP BY output_idx
{% endmacro %}


{% macro update_biases(bias_table, gradient_table, learning_rate=0.01) %}
-- Updates biases using gradient descent
SELECT 
    b.output_idx,
    b.bias_value - {{ learning_rate }} * COALESCE(g.bias_gradient, 0.0) AS bias_value
FROM {{ ref(bias_table) }} b
LEFT JOIN {{ gradient_table }} g ON b.output_idx = g.output_idx
{% endmacro %}


/* -----------------------------------------------------------------------------
   CROSS-ENTROPY LOSS COMPUTATION
   
   Loss = -sum(y_true * log(y_pred)) / batch_size
   For single-class: Loss = -log(p_correct)
   ----------------------------------------------------------------------------- */
{% macro compute_cross_entropy_loss() %}
-- Computes average cross-entropy loss over the batch
WITH correct_probs AS (
    SELECT 
        p.sample_id,
        p.probability
    FROM {{ ref('forward_pass_softmax') }} p
    INNER JOIN {{ ref('true_labels') }} t 
        ON p.sample_id = t.sample_id AND p.class_idx = t.true_label
)
SELECT 
    -AVG(LN(GREATEST(probability, 1e-10))) AS loss,
    COUNT(*) AS batch_size
FROM correct_probs
{% endmacro %}


/* -----------------------------------------------------------------------------
   HELPER: Convert neuron activations to array format
   ----------------------------------------------------------------------------- */
{% macro activations_to_array(input_cte) %}
SELECT 
    sample_id,
    to_json(
        transform(
            array_sort(collect_list(struct(neuron_idx AS idx, activation AS val))),
            x -> x.val
        )
    ) AS activations
FROM {{ input_cte }}
GROUP BY sample_id
{% endmacro %}

