# Relative valuation workflow

Run the project as one pipeline. Each step writes the file used by the next step.

CSV files are written with `utf-8-sig` so Chinese company names open correctly in Excel.

## Main scripts

```text
fetch_rqdatac_panel_data.py
step1_comparable_companies.py
step2_multiple_selection.py
step3_market_implied_multiple_prediction.py
company_valuation_cli.py
valuation_api_server.py
```

## Output folders

```text
outputs/raw_rqdatac_database/
outputs/calculated_feature_database/
```

`outputs/raw_rqdatac_database/` stores raw RQData files and market labels.

`outputs/calculated_feature_database/` stores calculated features, Step 1 outputs, Step 2 outputs, Step 3 predictions, and model metrics.

## Full run

Default date range: `2020q1` to `2025q4`.

```text
python3 fetch_rqdatac_panel_data.py
python3 step1_comparable_companies.py
python3 step2_multiple_selection.py
python3 step3_market_implied_multiple_prediction.py
```

For another date range, pass the same range to the fetch, Step 1, and Step 3 scripts:

```text
python3 fetch_rqdatac_panel_data.py --start-quarter 2021q1 --end-quarter 2024q4
python3 step1_comparable_companies.py --start-quarter 2021q1 --end-quarter 2024q4
python3 step2_multiple_selection.py
python3 step3_market_implied_multiple_prediction.py --start-quarter 2021q1 --end-quarter 2024q4
```

## Step outputs

### Fetch

Input: RQData.

Outputs:

```text
outputs/raw_rqdatac_database/raw_rqdatac_database_2020q1_2025q4.csv
outputs/raw_rqdatac_database/raw_market_labels_2020q1_2025q4.csv
outputs/calculated_feature_database/calculated_feature_input_2020q1_2025q4.csv
```

### Step 1

Script:

```text
python3 step1_comparable_companies.py
```

Input:

```text
outputs/calculated_feature_database/calculated_feature_input_2020q1_2025q4.csv
```

Main output for Step 2:

```text
outputs/calculated_feature_database/step1_to_step2_input.csv
```

Other Step 1 outputs:

```text
outputs/calculated_feature_database/step1_clean_universe.csv
outputs/calculated_feature_database/step1_cluster_assignments.csv
outputs/calculated_feature_database/step1_comparable_pairs.csv
outputs/calculated_feature_database/step1_comparable_summary.csv
outputs/calculated_feature_database/step1_run_summary.txt
```

### Step 2

Script:

```text
python3 step2_multiple_selection.py
```

Input:

```text
outputs/calculated_feature_database/step1_to_step2_input.csv
```

Output for Step 3:

```text
outputs/calculated_feature_database/step2_selected_multiples.csv
```

Step 2 adds the selected valuation multiple and the selection reason.

### Step 3

Script:

```text
python3 step3_market_implied_multiple_prediction.py
```

Inputs:

```text
outputs/calculated_feature_database/step2_selected_multiples.csv
outputs/raw_rqdatac_database/raw_market_labels_2020q1_2025q4.csv
```

Outputs:

```text
outputs/calculated_feature_database/step3_market_implied_multiple_predictions.csv
outputs/calculated_feature_database/step3_model_metrics.csv
outputs/calculated_feature_database/step3_run_summary.txt
```

Step 3 writes the final fair multiple, fair value, market comparison, confidence fields, and valuation signal.

## Lookup tools

Use the CLI after Step 3 has finished:

```text
python3 company_valuation_cli.py 平安银行 --quarter 2025q4
python3 company_valuation_cli.py 000001.XSHE --quarter latest
```

The CLI reads:

```text
outputs/calculated_feature_database/step3_market_implied_multiple_predictions.csv
outputs/calculated_feature_database/step3_model_metrics.csv
```

Start the local API and dashboard server with:

```text
python3 valuation_api_server.py
```

Default address:

```text
http://127.0.0.1:8765/
```

The server reads the same Step 3 prediction and metrics files as the CLI.
