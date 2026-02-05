/*
    VERIFICATION QUERIES FOR SQL NEURAL NETWORK
    
    Run these manually in Databricks SQL Editor after dbt seed and dbt run
    to verify that the neural network is working correctly.
*/

-- ===========================================================================
-- 1. VERIFY SEEDS LOADED CORRECTLY
-- ===========================================================================

-- Check row counts for all seed tables
SELECT COUNT(*) as count, 'fc1_bias' as table_name 
FROM workspace.neural_network.fc1_bias
UNION ALL
SELECT COUNT(*), 'fc2_bias' FROM workspace.neural_network.fc2_bias
UNION ALL
SELECT COUNT(*), 'fc3_bias' FROM workspace.neural_network.fc3_bias
UNION ALL
SELECT COUNT(*), 'fc1_weights_databricks' FROM workspace.neural_network.fc1_weights_databricks
UNION ALL
SELECT COUNT(*), 'fc2_weights_databricks' FROM workspace.neural_network.fc2_weights_databricks
UNION ALL
SELECT COUNT(*), 'fc3_weights_databricks' FROM workspace.neural_network.fc3_weights_databricks
UNION ALL
SELECT COUNT(*), 'input_images' FROM workspace.neural_network.input_images
UNION ALL
SELECT COUNT(*), 'true_labels' FROM workspace.neural_network.true_labels
ORDER BY table_name;

/* Expected counts:
   fc1_bias: 128
   fc1_weights_databricks: 128
   fc2_bias: 64
   fc2_weights_databricks: 64
   fc3_bias: 10
   fc3_weights_databricks: 10
   input_images: 7840 (10 samples × 784 pixels)
   true_labels: 10
*/


-- Verify weight arrays look correct (should have long strings)
SELECT 
    output_idx, 
    LENGTH(weights) as weight_string_length,
    SUBSTRING(weights, 1, 50) as weight_preview
FROM workspace.neural_network.fc1_weights_databricks
LIMIT 5;


-- Check bias values are reasonable floats
SELECT 
    output_idx,
    bias_value,
    ABS(bias_value) as abs_value
FROM workspace.neural_network.fc1_bias
ORDER BY output_idx
LIMIT 10;


-- Check input images have normalized pixel values (should be between -1 and 1)
SELECT 
    sample_id,
    MIN(pixel_value) as min_pixel,
    MAX(pixel_value) as max_pixel,
    AVG(pixel_value) as avg_pixel
FROM workspace.neural_network.input_images
GROUP BY sample_id
ORDER BY sample_id;


-- ===========================================================================
-- 2. VERIFY FORWARD PASS OUTPUTS (After dbt run)
-- ===========================================================================

-- Check forward pass produces 10 outputs (classes) per sample
SELECT 
    sample_id,
    COUNT(*) as num_classes,
    ROUND(MIN(logit), 4) as min_logit,
    ROUND(MAX(logit), 4) as max_logit,
    ROUND(AVG(logit), 4) as avg_logit
FROM workspace.neural_network.forward_pass_databricks
GROUP BY sample_id
ORDER BY sample_id;
-- Should have exactly 10 classes per sample


-- View all logits for first sample
SELECT 
    sample_id,
    class_idx,
    ROUND(logit, 4) as logit
FROM workspace.neural_network.forward_pass_databricks
WHERE sample_id = 0
ORDER BY class_idx;


-- ===========================================================================
-- 3. VERIFY PREDICTIONS
-- ===========================================================================

-- View all predictions
SELECT 
    sample_id,
    predicted_digit,
    ROUND(logit_score, 4) as logit,
    ROUND(softmax_probability, 4) as probability
FROM workspace.neural_network.predictions_databricks
ORDER BY sample_id;


-- Compare predictions to ground truth
SELECT 
    v.sample_id,
    v.true_label,
    v.databricks_prediction,
    v.databricks_correct,
    ROUND(v.databricks_logit, 4) as logit_score
FROM workspace.neural_network.validation v
ORDER BY v.sample_id;


-- Overall accuracy
SELECT 
    COUNT(*) as total_samples,
    SUM(databricks_correct) as correct_predictions,
    ROUND(AVG(databricks_correct) * 100, 2) as accuracy_percentage
FROM workspace.neural_network.validation;
-- Expected: Should be >90% for a trained MNIST model


-- ===========================================================================
-- 4. DETAILED ANALYSIS
-- ===========================================================================

-- Show which samples were predicted correctly vs incorrectly
SELECT 
    CASE WHEN databricks_correct = 1 THEN 'Correct' ELSE 'Incorrect' END as result,
    COUNT(*) as count
FROM workspace.neural_network.validation
GROUP BY databricks_correct;


-- Show confusion matrix (what digits get misclassified as what)
SELECT 
    true_label,
    databricks_prediction,
    COUNT(*) as count
FROM workspace.neural_network.validation
WHERE databricks_correct = 0
GROUP BY true_label, databricks_prediction
ORDER BY true_label, databricks_prediction;


-- For each digit, show accuracy
SELECT 
    true_label as digit,
    COUNT(*) as total_samples,
    SUM(databricks_correct) as correct,
    ROUND(AVG(databricks_correct) * 100, 2) as accuracy_pct
FROM workspace.neural_network.validation
GROUP BY true_label
ORDER BY true_label;


-- Show the hardest samples (where the model had low confidence but was still correct)
SELECT 
    v.sample_id,
    v.true_label,
    v.databricks_prediction,
    ROUND(v.databricks_logit, 4) as logit,
    v.databricks_correct
FROM workspace.neural_network.validation v
WHERE v.databricks_correct = 1
ORDER BY v.databricks_logit ASC
LIMIT 5;


-- Show the most confident wrong predictions
SELECT 
    v.sample_id,
    v.true_label,
    v.databricks_prediction,
    ROUND(v.databricks_logit, 4) as logit,
    v.databricks_correct
FROM workspace.neural_network.validation v
WHERE v.databricks_correct = 0
ORDER BY v.databricks_logit DESC
LIMIT 5;

