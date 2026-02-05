#!/usr/bin/env python3
"""
ULTRA-FAST Mini-Batch SGD Training

Optimizations:
1. Pre-exploded weights (no JSON parsing)
2. Direct SQL execution (no dbt overhead)
3. Minimal data movement
4. Efficient weight updates
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
import math
from databricks import sql as databricks_sql
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# FAST Training SQL - uses pre-exploded weights (no JSON parsing!)
TRAINING_SQL = """
-- Batch {batch_id}, LR={learning_rate}

WITH 
-- Batch selection
batch_samples AS (
    SELECT sample_id, ROW_NUMBER() OVER (ORDER BY sample_id) - 1 AS row_num
    FROM (SELECT DISTINCT sample_id FROM {schema}.input_images)
),
current_batch AS (
    SELECT sample_id FROM batch_samples 
    WHERE row_num >= {batch_start} AND row_num < {batch_end}
),
n_samples AS (SELECT COUNT(*) AS n FROM current_batch),

-- Input pixels (already in good format)
inputs AS (
    SELECT i.sample_id, i.pixel_idx AS input_idx, i.pixel_value AS val
    FROM {schema}.input_images i
    INNER JOIN current_batch b ON i.sample_id = b.sample_id
),

-- Layer 1 forward: z1 = x @ W1.T + b1, a1 = ReLU(z1)
l1_pre AS (
    SELECT 
        i.sample_id,
        w.output_idx AS neuron_idx,
        SUM(i.val * w.weight_value) + b.bias_value AS z
    FROM inputs i
    INNER JOIN {schema}.weights_fc1_fast w ON i.input_idx = w.input_idx
    INNER JOIN {schema}.biases_fc1 b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
l1_act AS (
    SELECT sample_id, neuron_idx, GREATEST(0.0, z) AS a, z
    FROM l1_pre
),

-- Layer 2 forward
l2_pre AS (
    SELECT 
        a.sample_id,
        w.output_idx AS neuron_idx,
        SUM(a.a * w.weight_value) + b.bias_value AS z
    FROM l1_act a
    INNER JOIN {schema}.weights_fc2_fast w ON a.neuron_idx = w.input_idx
    INNER JOIN {schema}.biases_fc2 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),
l2_act AS (
    SELECT sample_id, neuron_idx, GREATEST(0.0, z) AS a, z
    FROM l2_pre
),

-- Layer 3 forward (logits)
l3_logits AS (
    SELECT 
        a.sample_id,
        w.output_idx AS class_idx,
        SUM(a.a * w.weight_value) + b.bias_value AS logit
    FROM l2_act a
    INNER JOIN {schema}.weights_fc3_fast w ON a.neuron_idx = w.input_idx
    INNER JOIN {schema}.biases_fc3 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),

-- Softmax
max_logits AS (SELECT sample_id, MAX(logit) AS max_l FROM l3_logits GROUP BY sample_id),
softmax AS (
    SELECT 
        l.sample_id, 
        l.class_idx, 
        EXP(l.logit - m.max_l) / SUM(EXP(l.logit - m.max_l)) OVER (PARTITION BY l.sample_id) AS prob
    FROM l3_logits l
    INNER JOIN max_logits m ON l.sample_id = m.sample_id
),

-- Labels for this batch
batch_labels AS (
    SELECT t.sample_id, t.true_label
    FROM {schema}.true_labels t
    INNER JOIN current_batch b ON t.sample_id = b.sample_id
),

-- Output gradient: dL/dz3 = softmax - one_hot
dz3 AS (
    SELECT 
        p.sample_id, 
        p.class_idx AS neuron_idx,
        p.prob - CASE WHEN p.class_idx = t.true_label THEN 1.0 ELSE 0.0 END AS grad
    FROM softmax p
    INNER JOIN batch_labels t ON p.sample_id = t.sample_id
),

-- Layer 3 gradients
grad_w3 AS (
    SELECT 
        g.neuron_idx AS output_idx,
        a.neuron_idx AS input_idx,
        SUM(g.grad * a.a) / ns.n AS grad
    FROM dz3 g
    INNER JOIN l2_act a ON g.sample_id = a.sample_id
    CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, a.neuron_idx, ns.n
),
grad_b3 AS (
    SELECT neuron_idx AS output_idx, SUM(grad) / ns.n AS grad
    FROM dz3 CROSS JOIN n_samples ns
    GROUP BY neuron_idx, ns.n
),

-- Backprop through L3
da2 AS (
    SELECT g.sample_id, w.input_idx AS neuron_idx, SUM(g.grad * w.weight_value) AS grad
    FROM dz3 g
    INNER JOIN {schema}.weights_fc3_fast w ON g.neuron_idx = w.output_idx
    GROUP BY g.sample_id, w.input_idx
),
dz2 AS (
    SELECT g.sample_id, g.neuron_idx, CASE WHEN a.z > 0 THEN g.grad ELSE 0.0 END AS grad
    FROM da2 g
    INNER JOIN l2_act a ON g.sample_id = a.sample_id AND g.neuron_idx = a.neuron_idx
),

-- Layer 2 gradients
grad_w2 AS (
    SELECT g.neuron_idx AS output_idx, a.neuron_idx AS input_idx, SUM(g.grad * a.a) / ns.n AS grad
    FROM dz2 g
    INNER JOIN l1_act a ON g.sample_id = a.sample_id
    CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, a.neuron_idx, ns.n
),
grad_b2 AS (
    SELECT neuron_idx AS output_idx, SUM(grad) / ns.n AS grad
    FROM dz2 CROSS JOIN n_samples ns
    GROUP BY neuron_idx, ns.n
),

-- Backprop through L2
da1 AS (
    SELECT g.sample_id, w.input_idx AS neuron_idx, SUM(g.grad * w.weight_value) AS grad
    FROM dz2 g
    INNER JOIN {schema}.weights_fc2_fast w ON g.neuron_idx = w.output_idx
    GROUP BY g.sample_id, w.input_idx
),
dz1 AS (
    SELECT g.sample_id, g.neuron_idx, CASE WHEN a.z > 0 THEN g.grad ELSE 0.0 END AS grad
    FROM da1 g
    INNER JOIN l1_act a ON g.sample_id = a.sample_id AND g.neuron_idx = a.neuron_idx
),

-- Layer 1 gradients
grad_w1 AS (
    SELECT g.neuron_idx AS output_idx, i.input_idx, SUM(g.grad * i.val) / ns.n AS grad
    FROM dz1 g
    INNER JOIN inputs i ON g.sample_id = i.sample_id
    CROSS JOIN n_samples ns
    GROUP BY g.neuron_idx, i.input_idx, ns.n
),
grad_b1 AS (
    SELECT neuron_idx AS output_idx, SUM(grad) / ns.n AS grad
    FROM dz1 CROSS JOIN n_samples ns
    GROUP BY neuron_idx, ns.n
),

-- Metrics
batch_loss AS (
    SELECT -AVG(LN(GREATEST(p.prob, 1e-10))) AS loss
    FROM softmax p
    INNER JOIN batch_labels t ON p.sample_id = t.sample_id AND p.class_idx = t.true_label
),
preds AS (
    SELECT sample_id, class_idx, ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY prob DESC) AS rn
    FROM softmax
),
batch_acc AS (
    SELECT CAST(SUM(CASE WHEN p.class_idx = t.true_label THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS acc
    FROM preds p
    INNER JOIN batch_labels t ON p.sample_id = t.sample_id
    WHERE p.rn = 1
)

-- Output gradients + metrics
SELECT 'grad_w1' AS typ, output_idx, input_idx, grad, NULL AS loss, NULL AS acc FROM grad_w1
UNION ALL SELECT 'grad_b1', output_idx, NULL, grad, NULL, NULL FROM grad_b1
UNION ALL SELECT 'grad_w2', output_idx, input_idx, grad, NULL, NULL FROM grad_w2
UNION ALL SELECT 'grad_b2', output_idx, NULL, grad, NULL, NULL FROM grad_b2
UNION ALL SELECT 'grad_w3', output_idx, input_idx, grad, NULL, NULL FROM grad_w3
UNION ALL SELECT 'grad_b3', output_idx, NULL, grad, NULL, NULL FROM grad_b3
UNION ALL SELECT 'metrics', NULL, NULL, NULL, loss, acc FROM batch_loss CROSS JOIN batch_acc
"""

# Weight update SQL - applies gradients
UPDATE_WEIGHTS_SQL = """
CREATE OR REPLACE TABLE {schema}.{table} AS
SELECT 
    w.output_idx,
    w.input_idx,
    w.weight_value - {lr} * COALESCE(g.grad, 0.0) AS weight_value
FROM {schema}.{table} w
LEFT JOIN __gradients__ g ON w.output_idx = g.output_idx AND w.input_idx = g.input_idx
"""

UPDATE_BIASES_SQL = """
CREATE OR REPLACE TABLE {schema}.{table} AS
SELECT 
    b.output_idx,
    b.bias_value - {lr} * COALESCE(g.grad, 0.0) AS bias_value
FROM {schema}.{table} b
LEFT JOIN __gradients__ g ON b.output_idx = g.output_idx
"""

VALIDATION_SQL = """
WITH 
inputs AS (
    SELECT sample_id, pixel_idx AS input_idx, pixel_value AS val
    FROM {schema}.validation_images
),
l1_act AS (
    SELECT i.sample_id, w.output_idx AS neuron_idx, GREATEST(0.0, SUM(i.val * w.weight_value) + b.bias_value) AS a
    FROM inputs i
    INNER JOIN {schema}.weights_fc1_fast w ON i.input_idx = w.input_idx
    INNER JOIN {schema}.biases_fc1 b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
l2_act AS (
    SELECT a.sample_id, w.output_idx AS neuron_idx, GREATEST(0.0, SUM(a.a * w.weight_value) + b.bias_value) AS a
    FROM l1_act a
    INNER JOIN {schema}.weights_fc2_fast w ON a.neuron_idx = w.input_idx
    INNER JOIN {schema}.biases_fc2 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),
l3_logits AS (
    SELECT a.sample_id, w.output_idx AS class_idx, SUM(a.a * w.weight_value) + b.bias_value AS logit
    FROM l2_act a
    INNER JOIN {schema}.weights_fc3_fast w ON a.neuron_idx = w.input_idx
    INNER JOIN {schema}.biases_fc3 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),
max_logits AS (SELECT sample_id, MAX(logit) AS m FROM l3_logits GROUP BY sample_id),
softmax AS (
    SELECT l.sample_id, l.class_idx, EXP(l.logit - m.m) / SUM(EXP(l.logit - m.m)) OVER (PARTITION BY l.sample_id) AS prob
    FROM l3_logits l INNER JOIN max_logits m ON l.sample_id = m.sample_id
),
preds AS (SELECT sample_id, class_idx, ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY prob DESC) AS rn FROM softmax),
acc AS (
    SELECT CAST(SUM(CASE WHEN p.class_idx = t.true_label THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS accuracy
    FROM preds p INNER JOIN {schema}.validation_labels t ON p.sample_id = t.sample_id WHERE p.rn = 1
),
loss AS (
    SELECT -AVG(LN(GREATEST(p.prob, 1e-10))) AS loss
    FROM softmax p INNER JOIN {schema}.validation_labels t ON p.sample_id = t.sample_id AND p.class_idx = t.true_label
)
SELECT loss, accuracy FROM loss CROSS JOIN acc
"""


def load_config():
    load_dotenv()
    host = os.getenv('DATABRICKS_HOST')
    http_path = os.getenv('DATABRICKS_HTTP_PATH')
    token = os.getenv('DATABRICKS_TOKEN')
    if not all([host, http_path, token]):
        import yaml
        p = Path(__file__).parent.parent / 'profiles.yml'
        if p.exists():
            cfg = yaml.safe_load(open(p)).get('sql_neural_network', {})
            t = cfg.get('target', 'dev')
            o = cfg.get('outputs', {}).get(t, {})
            host, http_path, token = host or o.get('host'), http_path or o.get('http_path'), token or o.get('token')
    return {'server_hostname': host, 'http_path': http_path, 'access_token': token}


def run_dbt(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode == 0


def train_batch(cursor, schema, batch_start, batch_end, batch_id, lr):
    """Run forward+backward pass, return gradients and metrics."""
    sql = TRAINING_SQL.format(
        schema=schema, batch_start=batch_start, batch_end=batch_end, 
        batch_id=batch_id, learning_rate=lr
    )
    cursor.execute(sql)
    rows = cursor.fetchall()
    
    grads = {'w1': [], 'b1': [], 'w2': [], 'b2': [], 'w3': [], 'b3': []}
    metrics = {}
    
    for row in rows:
        typ = row[0]
        if typ == 'metrics':
            metrics = {'loss': row[4], 'accuracy': row[5]}
        elif typ == 'grad_w1':
            grads['w1'].append((row[1], row[2], row[3]))
        elif typ == 'grad_b1':
            grads['b1'].append((row[1], row[3]))
        elif typ == 'grad_w2':
            grads['w2'].append((row[1], row[2], row[3]))
        elif typ == 'grad_b2':
            grads['b2'].append((row[1], row[3]))
        elif typ == 'grad_w3':
            grads['w3'].append((row[1], row[2], row[3]))
        elif typ == 'grad_b3':
            grads['b3'].append((row[1], row[3]))
    
    return grads, metrics


def apply_gradients(cursor, schema, grads, lr):
    """Apply gradients to update weights."""
    # Weight updates
    for layer, table in [('w1', 'weights_fc1_fast'), ('w2', 'weights_fc2_fast'), ('w3', 'weights_fc3_fast')]:
        if grads[layer]:
            # Create temp table with gradients
            values = ', '.join([f"({int(o)}, {int(i)}, {g})" for o, i, g in grads[layer]])
            cursor.execute(f"CREATE OR REPLACE TEMP VIEW grad_temp AS SELECT * FROM VALUES {values} AS t(output_idx, input_idx, grad)")
            cursor.execute(f"""
                CREATE OR REPLACE TABLE {schema}.{table} AS
                SELECT w.output_idx, w.input_idx, w.weight_value - {lr} * COALESCE(g.grad, 0.0) AS weight_value
                FROM {schema}.{table} w
                LEFT JOIN grad_temp g ON w.output_idx = g.output_idx AND w.input_idx = g.input_idx
            """)
    
    # Bias updates
    for layer, table in [('b1', 'biases_fc1'), ('b2', 'biases_fc2'), ('b3', 'biases_fc3')]:
        if grads[layer]:
            values = ', '.join([f"({int(o)}, {g})" for o, g in grads[layer]])
            cursor.execute(f"CREATE OR REPLACE TEMP VIEW grad_temp AS SELECT * FROM VALUES {values} AS t(output_idx, grad)")
            cursor.execute(f"""
                CREATE OR REPLACE TABLE {schema}.{table} AS
                SELECT b.output_idx, b.bias_value - {lr} * COALESCE(g.grad, 0.0) AS bias_value
                FROM {schema}.{table} b
                LEFT JOIN grad_temp g ON b.output_idx = g.output_idx
            """)


def validate(cursor, schema):
    cursor.execute(VALIDATION_SQL.format(schema=schema))
    row = cursor.fetchone()
    return {'loss': row[0], 'accuracy': row[1]}


def plot_curves(history, path):
    if len(history) < 2: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [h['epoch'] for h in history]
    ax1.plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train')
    if history[0].get('val_loss'): ax1.plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, [h['train_accuracy']*100 for h in history], 'b-', label='Train')
    if history[0].get('val_accuracy'): ax2.plot(epochs, [h['val_accuracy']*100 for h in history], 'r-', label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0,100)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--catalog', type=str, default='workspace')
    parser.add_argument('--schema', type=str, default='neural_network_training')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--validate-every', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.parent
    seeds_dir = project_dir / 'seeds'
    schema = f"{args.catalog}.{args.schema}"
    
    print("=" * 70)
    print("  ⚡ ULTRA-FAST SQL Neural Network Training")
    print("=" * 70)
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Mode:           Pre-exploded weights + Direct SQL")
    print("=" * 70)
    
    # Check for fast weight files
    if not (seeds_dir / 'weights_fc1_fast.csv').exists() or args.init:
        print("\nInitializing weights (fast format)...")
        subprocess.run(['python', Path(__file__).parent / 'init_weights_fast.py'], check=True)
    
    # Load seeds
    print("\nLoading data...")
    if not run_dbt(['dbt', 'seed', '--full-refresh'], project_dir):
        print("Failed to load seeds"); return
    
    # Connect
    conn = databricks_sql.connect(**load_config())
    cursor = conn.cursor()
    
    try:
        # Count samples
        cursor.execute(f"SELECT COUNT(DISTINCT sample_id) FROM {schema}.input_images")
        n_samples = cursor.fetchone()[0]
        n_batches = math.ceil(n_samples / args.batch_size)
        
        print(f"\n  Samples: {n_samples}, Batches: {n_batches}")
        
        history = []
        best_val = float('inf')
        
        print("\n" + "─" * 70)
        
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            losses, accs = [], []
            
            # Batch order
            batch_order = list(range(n_batches))
            if args.shuffle:
                np.random.shuffle(batch_order)
            
            print(f"  Epoch {epoch}/{args.epochs}: ", end='', flush=True)
            
            for batch_idx in batch_order:
                batch_start = batch_idx * args.batch_size
                batch_end = min(batch_start + args.batch_size, n_samples)
                
                try:
                    grads, metrics = train_batch(cursor, schema, batch_start, batch_end, batch_idx, args.learning_rate)
                    losses.append(metrics['loss'])
                    accs.append(metrics['accuracy'])
                    apply_gradients(cursor, schema, grads, args.learning_rate)
                    print('.', end='', flush=True)
                except Exception as e:
                    print('x', end='', flush=True)
                    if args.verbose: print(f"\n    Error: {e}")
            
            epoch_time = time.time() - t0
            avg_loss = np.mean(losses) if losses else 0
            avg_acc = np.mean(accs) if accs else 0
            
            # Validation
            val_loss, val_acc = None, None
            if args.validate_every and epoch % args.validate_every == 0:
                try:
                    v = validate(cursor, schema)
                    val_loss, val_acc = v['loss'], v['accuracy']
                except: pass
            
            marker = " ✓" if val_loss and val_loss < best_val else ""
            if val_loss and val_loss < best_val: best_val = val_loss
            
            val_str = f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}%" if val_loss else ""
            print(f" loss={avg_loss:.4f} acc={avg_acc*100:.1f}% {val_str} ({epoch_time:.1f}s){marker}")
            
            history.append({
                'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc,
                'val_loss': val_loss, 'val_accuracy': val_acc, 'time': epoch_time
            })
        
        print("─" * 70)
        if history:
            total = sum(h['time'] for h in history)
            print(f"\n  Total: {total:.1f}s ({total/len(history):.1f}s/epoch)")
            pd.DataFrame(history).to_csv(project_dir / 'training_history.csv', index=False)
            plot_curves(history, project_dir / 'training_curves.png')
    
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
