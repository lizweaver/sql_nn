# SQL Neural Network

This project implements a trained MNIST neural network using SQL and dbt, with support for Databricks, Snowflake, and database-agnostic wide-format implementations.

## Architecture

The neural network has the following architecture:
- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1**: 128 neurons + ReLU activation
- **Hidden Layer 2**: 64 neurons + ReLU activation
- **Output Layer**: 10 neurons (digit classes 0-9)

## Project Structure

```
architecture/
├── dbt_project.yml           # dbt project configuration
├── profiles.yml               # Database connection settings
├── models/
│   ├── databricks/           # Databricks implementation (Spark SQL)
│   │   ├── forward_pass_databricks.sql
│   │   ├── predictions_databricks.sql
│   │   └── schema.yml
│   ├── snowflake/            # Snowflake implementation
│   │   ├── forward_pass_snowflake.sql
│   │   ├── predictions_snowflake.sql
│   │   └── schema.yml
│   ├── wide/                 # Database-agnostic wide format
│   │   ├── forward_pass_wide.sql
│   │   ├── predictions_wide.sql
│   │   └── schema.yml
│   └── validation.sql        # Compare implementations
├── macros/
│   ├── nn_databricks.sql     # Databricks array macros
│   ├── nn_snowflake.sql      # Snowflake array macros
│   └── nn_wide.sql           # Wide format Jinja2 macros
├── seeds/                    # Generated CSV files with weights
├── scripts/
│   ├── export_weights.py     # Export PyTorch weights to CSV
│   └── verify_predictions.py # Compare SQL vs PyTorch predictions
└── verification_queries.sql  # Manual verification queries
```

## Setup

### 1. Install Dependencies

```bash
# Core packages
pip install dbt-databricks torch torchvision numpy pandas

# For verification (optional)
pip install databricks-sql-connector
```

### 2. Configure Database Connection

Set environment variables:
```bash
export DATABRICKS_HOST="your-workspace.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxxx"
export DATABRICKS_TOKEN="dapi..."
```

Copy `profiles.yml` to `~/.dbt/profiles.yml` or keep it in the project directory.

### 3. Export Model Weights

```bash
cd architecture
python scripts/export_weights.py
```

This creates CSV files in the `seeds/` directory with:
- Weight matrices (3 formats: row, Snowflake array, Databricks array)
- Bias vectors
- Sample MNIST images for testing

### 4. Load Seeds and Run Models

```bash
# Load only Databricks seeds (faster)
dbt seed --select fc1_weights_databricks fc2_weights_databricks fc3_weights_databricks fc1_bias fc2_bias fc3_bias input_images true_labels

# Or load all seeds (if using multiple implementations)
dbt seed

# Run the forward pass and predictions
dbt run --select tag:databricks

# Run validation to compare with ground truth
dbt run --select validation
```

## Verification

### Option 1: SQL Queries (Manual)

Open `verification_queries.sql` and run queries in Databricks SQL Editor to check:
- Seed data loaded correctly
- Forward pass produces expected outputs
- Predictions match ground truth
- Overall accuracy

### Option 2: Python Verification Script (Automated)

Compare SQL predictions directly with PyTorch:

```bash
python scripts/verify_predictions.py
```

This will:
1. Load the PyTorch model
2. Run inference on test samples
3. Fetch predictions from Databricks
4. Compare predictions and logits
5. Show match rate and differences

Expected output:
```
✅ SUCCESS: Databricks SQL implementation matches PyTorch!
Total samples compared: 10
Predictions match: 10/10 (100.0%)
PyTorch accuracy: 90.0%
Databricks SQL accuracy: 90.0%
```

## Implementation Details

### Databricks (Spark SQL)
- Uses `posexplode()` and `collect_list()` for array operations
- Stores weights as array literals: `array(0.1, 0.2, ...)`
- Most efficient for cloud data warehouses

### Snowflake
- Uses `FLATTEN()` and `ARRAY_AGG()` for array operations
- Stores weights as JSON arrays
- Similar performance to Databricks

### Wide Format
- Database-agnostic (works on any SQL database)
- Uses Jinja2 to generate explicit column-wise operations
- Generates verbose SQL but highly portable
- Best for databases without array support

## Query Results

After running `dbt run`, you can query results in Databricks:

```sql
-- View predictions
SELECT * FROM workspace.neural_network.predictions_databricks;

-- Check accuracy
SELECT 
    COUNT(*) as total,
    SUM(databricks_correct) as correct,
    ROUND(AVG(databricks_correct) * 100, 2) as accuracy_pct
FROM workspace.neural_network.validation;
```

## Performance Notes

- **Databricks/Snowflake**: Array format is most efficient (~128 rows per layer)
- **Wide format**: Row format requires 100K+ rows for fc1 layer alone
- **Seed time**: Array format seeds load in ~30 seconds vs ~5 minutes for row format

## Troubleshooting

### "Catalog not found"
Make sure to use the correct catalog name in `profiles.yml`. Check available catalogs:
```sql
SHOW CATALOGS;
```

### "VARCHAR requires length"
Use `string` type instead of `varchar` in `dbt_project.yml` for Databricks.

### Predictions don't match PyTorch
Small differences (< 0.01 in logits) are normal due to floating point precision. If predictions differ significantly, check:
1. Seeds loaded completely (`dbt seed` finished without errors)
2. Model file (`model.pth`) matches the weights being used
3. Input normalization matches PyTorch (mean=0.5, std=0.5)

## License

This is a demonstration project for educational purposes.

