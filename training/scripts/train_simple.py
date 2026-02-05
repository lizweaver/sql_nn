#!/usr/bin/env python3
"""
Simple 2-layer network training (784 → 32 → 10)

Much faster than 3-layer network due to:
- 4x fewer parameters
- 2x fewer layer computations
- Simpler SQL queries
"""

import subprocess
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


# Simple 2-layer training SQL
TRAINING_SQL = """
WITH 
-- Batch selection
samples AS (
    SELECT sample_id, ROW_NUMBER() OVER (ORDER BY sample_id) - 1 AS rn
    FROM (SELECT DISTINCT sample_id FROM {schema}.input_images)
),
batch AS (SELECT sample_id FROM samples WHERE rn >= {batch_start} AND rn < {batch_end}),
n AS (SELECT COUNT(*) AS n FROM batch),

-- Inputs for this batch
inputs AS (
    SELECT i.sample_id, i.pixel_idx, i.pixel_value
    FROM {schema}.input_images i
    INNER JOIN batch b ON i.sample_id = b.sample_id
),

-- Layer 1: 784 → 32 with ReLU
w1 AS (
    SELECT output_idx, idx AS input_idx, val AS w
    FROM {schema}.weights_fc1
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
z1 AS (
    SELECT i.sample_id, w.output_idx AS idx, SUM(i.pixel_value * w.w) + b.bias_value AS z
    FROM inputs i
    INNER JOIN w1 w ON i.pixel_idx = w.input_idx
    INNER JOIN {schema}.biases_fc1 b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
a1 AS (SELECT sample_id, idx, GREATEST(0.0, z) AS a, z FROM z1),

-- Layer 2: 32 → 10 (output)
w2 AS (
    SELECT output_idx, idx AS input_idx, val AS w
    FROM {schema}.weights_fc2
    LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val
),
logits AS (
    SELECT a.sample_id, w.output_idx AS class_idx, SUM(a.a * w.w) + b.bias_value AS logit
    FROM a1 a
    INNER JOIN w2 w ON a.idx = w.input_idx
    INNER JOIN {schema}.biases_fc2 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),

-- Softmax
mx AS (SELECT sample_id, MAX(logit) AS m FROM logits GROUP BY sample_id),
softmax AS (
    SELECT l.sample_id, l.class_idx, 
           EXP(l.logit - mx.m) / SUM(EXP(l.logit - mx.m)) OVER (PARTITION BY l.sample_id) AS p
    FROM logits l INNER JOIN mx ON l.sample_id = mx.sample_id
),

-- Labels
labels AS (
    SELECT t.sample_id, t.true_label
    FROM {schema}.true_labels t INNER JOIN batch b ON t.sample_id = b.sample_id
),

-- Backward: dL/dz2 = softmax - onehot
dz2 AS (
    SELECT s.sample_id, s.class_idx AS idx, s.p - CASE WHEN s.class_idx = l.true_label THEN 1.0 ELSE 0.0 END AS g
    FROM softmax s INNER JOIN labels l ON s.sample_id = l.sample_id
),

-- Gradients for W2, b2
gw2 AS (
    SELECT d.idx AS out_idx, a.idx AS in_idx, SUM(d.g * a.a) / n.n AS grad
    FROM dz2 d INNER JOIN a1 a ON d.sample_id = a.sample_id CROSS JOIN n
    GROUP BY d.idx, a.idx, n.n
),
gb2 AS (SELECT idx AS out_idx, SUM(g) / n.n AS grad FROM dz2 CROSS JOIN n GROUP BY idx, n.n),

-- Backprop through layer 2
da1 AS (
    SELECT d.sample_id, w.input_idx AS idx, SUM(d.g * w.w) AS g
    FROM dz2 d INNER JOIN w2 w ON d.idx = w.output_idx
    GROUP BY d.sample_id, w.input_idx
),
dz1 AS (
    SELECT d.sample_id, d.idx, CASE WHEN a.z > 0 THEN d.g ELSE 0.0 END AS g
    FROM da1 d INNER JOIN a1 a ON d.sample_id = a.sample_id AND d.idx = a.idx
),

-- Gradients for W1, b1
gw1 AS (
    SELECT d.idx AS out_idx, i.pixel_idx AS in_idx, SUM(d.g * i.pixel_value) / n.n AS grad
    FROM dz1 d INNER JOIN inputs i ON d.sample_id = i.sample_id CROSS JOIN n
    GROUP BY d.idx, i.pixel_idx, n.n
),
gb1 AS (SELECT idx AS out_idx, SUM(g) / n.n AS grad FROM dz1 CROSS JOIN n GROUP BY idx, n.n),

-- Metrics
loss AS (
    SELECT -AVG(LN(GREATEST(s.p, 1e-10))) AS loss
    FROM softmax s INNER JOIN labels l ON s.sample_id = l.sample_id AND s.class_idx = l.true_label
),
preds AS (SELECT sample_id, class_idx, ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY p DESC) AS rn FROM softmax),
acc AS (
    SELECT CAST(SUM(CASE WHEN p.class_idx = l.true_label THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS acc
    FROM preds p INNER JOIN labels l ON p.sample_id = l.sample_id WHERE p.rn = 1
)

SELECT 'gw1' AS t, out_idx, in_idx, grad, NULL AS loss, NULL AS acc FROM gw1
UNION ALL SELECT 'gb1', out_idx, NULL, grad, NULL, NULL FROM gb1
UNION ALL SELECT 'gw2', out_idx, in_idx, grad, NULL, NULL FROM gw2
UNION ALL SELECT 'gb2', out_idx, NULL, grad, NULL, NULL FROM gb2
UNION ALL SELECT 'metrics', NULL, NULL, NULL, loss, acc FROM loss CROSS JOIN acc
"""

VALIDATION_SQL = """
WITH 
inputs AS (SELECT sample_id, pixel_idx, pixel_value FROM {schema}.validation_images),
w1 AS (SELECT output_idx, idx AS input_idx, val AS w FROM {schema}.weights_fc1 LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val),
a1 AS (
    SELECT i.sample_id, w.output_idx AS idx, GREATEST(0.0, SUM(i.pixel_value * w.w) + b.bias_value) AS a
    FROM inputs i INNER JOIN w1 w ON i.pixel_idx = w.input_idx INNER JOIN {schema}.biases_fc1 b ON w.output_idx = b.output_idx
    GROUP BY i.sample_id, w.output_idx, b.bias_value
),
w2 AS (SELECT output_idx, idx AS input_idx, val AS w FROM {schema}.weights_fc2 LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val),
logits AS (
    SELECT a.sample_id, w.output_idx AS class_idx, SUM(a.a * w.w) + b.bias_value AS logit
    FROM a1 a INNER JOIN w2 w ON a.idx = w.input_idx INNER JOIN {schema}.biases_fc2 b ON w.output_idx = b.output_idx
    GROUP BY a.sample_id, w.output_idx, b.bias_value
),
mx AS (SELECT sample_id, MAX(logit) AS m FROM logits GROUP BY sample_id),
softmax AS (
    SELECT l.sample_id, l.class_idx, EXP(l.logit - mx.m) / SUM(EXP(l.logit - mx.m)) OVER (PARTITION BY l.sample_id) AS p
    FROM logits l INNER JOIN mx ON l.sample_id = mx.sample_id
),
preds AS (SELECT sample_id, class_idx, ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY p DESC) AS rn FROM softmax),
acc AS (
    SELECT CAST(SUM(CASE WHEN p.class_idx = t.true_label THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS acc
    FROM preds p INNER JOIN {schema}.validation_labels t ON p.sample_id = t.sample_id WHERE p.rn = 1
),
loss AS (
    SELECT -AVG(LN(GREATEST(s.p, 1e-10))) AS loss
    FROM softmax s INNER JOIN {schema}.validation_labels t ON s.sample_id = t.sample_id AND s.class_idx = t.true_label
)
SELECT loss, acc FROM loss CROSS JOIN acc
"""


def load_config():
    load_dotenv()
    h, p, t = os.getenv('DATABRICKS_HOST'), os.getenv('DATABRICKS_HTTP_PATH'), os.getenv('DATABRICKS_TOKEN')
    if not all([h, p, t]):
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'profiles.yml'))
        o = cfg.get('sql_neural_network', {}).get('outputs', {}).get(cfg.get('sql_neural_network', {}).get('target', 'dev'), {})
        h, p, t = h or o.get('host'), p or o.get('http_path'), t or o.get('token')
    return {'server_hostname': h, 'http_path': p, 'access_token': t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)  # Default to full batch
    parser.add_argument('--learning-rate', type=float, default=0.1)  # Higher LR for small network
    parser.add_argument('--catalog', type=str, default='workspace')
    parser.add_argument('--schema', type=str, default='neural_network_training')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--validate-every', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.parent
    schema = f"{args.catalog}.{args.schema}"
    
    print("=" * 60)
    print("  Simple 2-Layer Network Training (784 → 32 → 10)")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print("=" * 60)
    
    # Init small network
    if args.init or not (project_dir / 'seeds' / 'weights_fc1.csv').exists():
        print("\nInitializing small network...")
        subprocess.run(['python', Path(__file__).parent / 'init_weights_small.py'], check=True)
    
    # Load seeds
    print("\nLoading data via dbt seed (this may take 1-2 min)...", flush=True)
    t_seed = time.time()
    result = subprocess.run(['dbt', 'seed', '--full-refresh'], cwd=project_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  dbt seed failed!")
        print(f"  stderr: {result.stderr[:300]}")
        print(f"  stdout: {result.stdout[-500:]}")
        return
    print(f"  Done ({time.time() - t_seed:.0f}s)")
    
    print("\nConnecting to Databricks...", flush=True)
    conn = databricks_sql.connect(**load_config())
    print("  Connected!")
    cursor = conn.cursor()
    
    try:
        print("\nCounting samples...", flush=True)
        cursor.execute(f"SELECT COUNT(DISTINCT sample_id) FROM {schema}.input_images")
        n_samples = cursor.fetchone()[0]
        n_batches = max(1, math.ceil(n_samples / args.batch_size))
        
        print(f"  Samples: {n_samples}, Batches/epoch: {n_batches}")
        history = []
        
        print("\n" + "─" * 60)
        print("  Starting training (first batch may take ~30-60s to warm up)...", flush=True)
        print("─" * 60)
        
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            losses, accs = [], []
            
            batch_order = list(range(n_batches))
            if args.shuffle: np.random.shuffle(batch_order)
            
            print(f"  Epoch {epoch}/{args.epochs}: ", end='', flush=True)
            
            for bi in batch_order:
                bs, be = bi * args.batch_size, min((bi + 1) * args.batch_size, n_samples)
                
                try:
                    cursor.execute(TRAINING_SQL.format(schema=schema, batch_start=bs, batch_end=be))
                    rows = cursor.fetchall()
                    
                    grads = {'w1': [], 'b1': [], 'w2': [], 'b2': []}
                    for r in rows:
                        if r[0] == 'metrics':
                            losses.append(r[4]); accs.append(r[5])
                        elif r[0] == 'gw1': grads['w1'].append((r[1], r[2], r[3]))
                        elif r[0] == 'gb1': grads['b1'].append((r[1], r[3]))
                        elif r[0] == 'gw2': grads['w2'].append((r[1], r[2], r[3]))
                        elif r[0] == 'gb2': grads['b2'].append((r[1], r[3]))
                    
                    # Update weights
                    lr = args.learning_rate
                    for layer, table in [('w1', 'weights_fc1'), ('w2', 'weights_fc2')]:
                        if grads[layer]:
                            vals = ', '.join([f"({int(o)}, {int(i)}, {g})" for o, i, g in grads[layer]])
                            cursor.execute(f"CREATE OR REPLACE TEMP VIEW gt AS SELECT * FROM VALUES {vals} AS t(o, i, g)")
                            cursor.execute(f"""
                                CREATE OR REPLACE TABLE {schema}.{table} AS
                                WITH old AS (SELECT output_idx, idx AS i, val AS w FROM {schema}.{table} LATERAL VIEW posexplode(from_json(weights, 'array<double>')) AS idx, val),
                                upd AS (SELECT o.output_idx, o.i, o.w - {lr} * COALESCE(g.g, 0.0) AS w FROM old o LEFT JOIN gt g ON o.output_idx = g.o AND o.i = g.i)
                                SELECT output_idx, to_json(transform(array_sort(collect_list(struct(i, w))), x -> x.w)) AS weights FROM upd GROUP BY output_idx
                            """)
                    
                    for layer, table in [('b1', 'biases_fc1'), ('b2', 'biases_fc2')]:
                        if grads[layer]:
                            vals = ', '.join([f"({int(o)}, {g})" for o, g in grads[layer]])
                            cursor.execute(f"CREATE OR REPLACE TEMP VIEW gt AS SELECT * FROM VALUES {vals} AS t(o, g)")
                            cursor.execute(f"""
                                CREATE OR REPLACE TABLE {schema}.{table} AS
                                SELECT b.output_idx, b.bias_value - {lr} * COALESCE(g.g, 0.0) AS bias_value
                                FROM {schema}.{table} b LEFT JOIN gt g ON b.output_idx = g.o
                            """)
                    
                    print('.', end='', flush=True)
                except Exception as e:
                    print('x', end='', flush=True)
                    if args.verbose: print(f"\n  {e}")
            
            epoch_time = time.time() - t0
            avg_loss = np.mean(losses) if losses else 0
            avg_acc = np.mean(accs) if accs else 0
            
            # Validation
            val_loss, val_acc = None, None
            if args.validate_every and epoch % args.validate_every == 0:
                try:
                    cursor.execute(VALIDATION_SQL.format(schema=schema))
                    r = cursor.fetchone()
                    val_loss, val_acc = r[0], r[1]
                except: pass
            
            val_str = f"val={val_loss:.3f}/{val_acc*100:.0f}%" if val_loss else ""
            print(f" loss={avg_loss:.3f} acc={avg_acc*100:.0f}% {val_str} ({epoch_time:.1f}s)")
            
            history.append({'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc,
                           'val_loss': val_loss, 'val_accuracy': val_acc, 'time': epoch_time})
        
        print("─" * 60)
        if history:
            total = sum(h['time'] for h in history)
            print(f"\n  Total: {total:.0f}s ({total/len(history):.0f}s/epoch)")
            pd.DataFrame(history).to_csv(project_dir / 'training_history.csv', index=False)
    
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()


