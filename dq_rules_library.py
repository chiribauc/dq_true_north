# dq_rules_library.py
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session, Row
from snowflake.snowpark.functions import col, count, when, lit, avg, stddev, lag, sum as snowpark_sum, dateadd, current_date, to_utc_timestamp, object_construct, current_timestamp, abs as snowpark_abs, to_date, concat_ws, coalesce, upper, round as snowpark_round
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.window import Window
import pandas as pd
import json
import logging
import re
import time # Import the time library for generating unique names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HELPER FUNCTION FOR LOGGING ---
def log_dq_result(
 session: Session, dq_results_table: str, rule_id: int, rule_name: str,
 segment_value: str, rule_type: str, logic_implementation: str,
 result_value: any, indicator: str, error_message: str
) -> None:
    """
    Logs the result of a data quality rule execution to the specified results table.
    """
    try:
        log_data = {
            "RULE_ID": rule_id,
            "RULE_NAME": rule_name,
            "SEGMENT_VALUE": segment_value,
            "RULE_TYPE": rule_type,
            "LOGIC_IMPLEMENTATION": logic_implementation,
            "RESULT_VALUE": str(result_value) if result_value is not None else None,
            "INDICATOR": indicator,
            "ERROR_MESSAGE": error_message
        }
        
        results_df = session.create_dataframe([log_data], schema=list(log_data.keys()))
        
        final_df = results_df.with_column("EXECUTION_TIMESTAMP", current_timestamp()).select(
            "RULE_ID", "RULE_NAME", "SEGMENT_VALUE", "RULE_TYPE", "LOGIC_IMPLEMENTATION",
            "EXECUTION_TIMESTAMP", "RESULT_VALUE", "INDICATOR", "ERROR_MESSAGE"
        )
        
        final_df.write.mode("append").save_as_table(dq_results_table)
        print(f"Inserted result for rule '{rule_name}' into '{dq_results_table}'.")
        
    except SnowparkSQLException as sqlex:
        print(f"Snowpark SQL error logging result for '{rule_name}': {sqlex}")
        logger.error(f"Snowpark SQL error logging result for '{rule_name}': {sqlex}", exc_info=True)
    except Exception as log_err:
        print(f"General error logging result for '{rule_name}': {log_err}")
        logger.error(f"General error logging result for '{rule_name}': {log_err}", exc_info=True)

# --- SQL QUERY RULE EXECUTION ---
def execute_sql_rule(session: Session, rule_row: Row, dq_results_table: str) -> None:
    """
    Executes a data quality rule defined by a SQL query.
    Handles UNIQUENESS, COMPLETENESS (with inline SQL), ROLLING_AVERAGE, and generic SQL rules.
    """
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_RULE')
    logic_definition = getattr(rule_row, 'LOGIC_DEFINITION', None)
    segment_value = getattr(rule_row, 'SEGMENT_VALUE', None)
    logic_implementation, rule_type = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SQL_QUERY'), getattr(rule_row, 'RULE_TYPE', 'UNKNOWN')
    logger.info(f"Executing SQL rule: {rule_name}")

    try:
        if rule_type == 'UNIQUENESS':
            logger.info(f"Detected UNIQUENESS rule. Extracting detailed duplicates for '{rule_name}'.")
            match = re.search(r'\((SELECT.*)\)\s+as\s+dupes', logic_definition, re.IGNORECASE | re.DOTALL)
            if not match:
                raise ValueError("Could not parse the inner query from the UNIQUENESS LOGIC_DEFINITION. Format: '... FROM (SELECT ...) as dupes'.")
            inner_query = match.group(1)
            
            duplicates_df = session.sql(inner_query)
            count_of_duplicate_groups = duplicates_df.count()

            if count_of_duplicate_groups > 0:
                logger.info(f"Found {count_of_duplicate_groups} duplicate groups. Saving details.")
                
                json_cols = duplicates_df.columns[:-1]
                count_col_name = duplicates_df.columns[-1]
                
                object_args = []
                for p_col in json_cols:
                    object_args.append(lit(p_col))
                    object_args.append(col(p_col))
                duplicate_values_obj = object_construct(*object_args)
                
                details_to_log = duplicates_df.with_column("RULE_NAME", lit(rule_name)) \
                    .with_column("DUPLICATE_VALUES", duplicate_values_obj) \
                    .withColumnRenamed(count_col_name, "DUPLICATE_COUNT")
                
                final_details_df = details_to_log.select("RULE_NAME", "DUPLICATE_VALUES", "DUPLICATE_COUNT") \
                    .with_column("EXECUTION_TIMESTAMP", current_timestamp())
                
                final_details_df.write.mode("append").save_as_table("DQ_DUPLICATE_DETAILS")
            
            final_indicator = "FAIL" if count_of_duplicate_groups > 0 else "PASS"
            log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, count_of_duplicate_groups, final_indicator, None)

        elif rule_type == 'COMPLETENESS' and logic_definition:
            logger.info(f"Executing COMPLETENESS rule with inline SQL: {rule_name}.")
            
            collected_results = session.sql(logic_definition).collect()
            
            final_indicator, result_value, error_message = "PASS", 0, None
            if collected_results and collected_results[0][0] is not None:
                result_value = collected_results[0][0]
                final_indicator = "FAIL" if result_value > 0 else "PASS"
            elif collected_results:
                error_message = f"Completeness check query for '{rule_name}' returned a NULL result."
                final_indicator = "FAIL"
                result_value = "NULL_RESULT"
            else:
                error_message = f"Completeness check query for '{rule_name}' returned no results."
                final_indicator = "ERROR"
                result_value = "NO_RESULTS"
            
            log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, result_value, final_indicator, error_message)
            
            if collected_results and len(collected_results[0]) == 4:
                try:
                    result_row = collected_results[0]
                    detail_log_df = session.create_dataframe([
                        (result_row[0], result_row[1], result_row[2], result_row[3])
                    ], schema=["RULE_NAME", "EXPECTED_DATE", "ACTUAL_MAX_DATE", "DAYS_MISSING"])
                    
                    final_detail_log_df = detail_log_df.with_column("EXECUTION_TIMESTAMP", current_timestamp())
                    final_detail_log_df.write.mode("append").save_as_table("DQ_DETAILS_COMPLETENESS")
                except Exception as detail_log_err:
                    error_message = f"Failed to log detailed completeness check for '{rule_name}': {detail_log_err}"
                    logger.error(error_message, exc_info=True)
        
        elif rule_type == 'ROLLING_AVERAGE':
            logger.info(f"Executing ROLLING_AVERAGE rule: {rule_name}.")
            
            try:
                params_str = getattr(rule_row, 'PARAMETERS', '{}') 
                params = json.loads(params_str)
                
                source_table = params.get("source_table")
                metric_column = params.get("metric_column")
                date_column = params.get("date_column")
                window_size = params.get("window_size", 60)
                segment_columns = params.get("segment_columns", [])
                
                if not all([source_table, metric_column, date_column]):
                    raise ValueError("Missing required parameters in PARAMETERS JSON (source_table, metric_column, date_column).")

                quoted_segment_cols = [f'"{c}"' for c in segment_columns]
                segment_partition_clause = f"PARTITION BY {', '.join(quoted_segment_cols)}" if segment_columns else ""
                
                rolling_avg_query = f"""
                WITH RollingAverages AS (
                    SELECT
                        *,
                        AVG("{metric_column}") OVER (
                            {segment_partition_clause}
                            ORDER BY "{date_column}"
                            ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW
                        ) AS ROLLING_AVERAGE_VALUE
                    FROM {source_table}
                )
                SELECT
                    '{rule_name}' AS RULE_NAME,
                    {', '.join(quoted_segment_cols) if segment_columns else "NULL as BRAND, NULL as PLATFORM"},
                    '{metric_column}' AS METRIC_COLUMN,
                    "{date_column}" AS CALCULATION_DATE,
                    ROLLING_AVERAGE_VALUE
                FROM RollingAverages
                QUALIFY ROW_NUMBER() OVER ({segment_partition_clause} ORDER BY "{date_column}" DESC) = 1
                """
                
                results_df = session.sql(rolling_avg_query)
                collected_results = results_df.collect()
                
                if not collected_results:
                    error_message = f"Rolling average calculation for '{rule_name}' returned no results."
                    log_dq_result(session, dq_results_table, rule_id, rule_name, "All Segments", rule_type, logic_implementation, "NO_RESULTS", "ERROR", error_message)

                else:
                    for result_row in collected_results:
                        try:
                            result_value = result_row["ROLLING_AVERAGE_VALUE"]
                            calculation_date = result_row["CALCULATION_DATE"]
                            
                            segment_value_dict = {col_name: result_row[col_name] for col_name in segment_columns}
                            segment_value_logged = json.dumps(segment_value_dict) if segment_value_dict else 'N/A'

                            detail_log_data = [(
                                result_row["RULE_NAME"],
                                segment_value_logged,
                                result_row["METRIC_COLUMN"],
                                result_value,
                                calculation_date
                            )]
                            
                            detail_schema = ["RULE_NAME", "SEGMENT_VALUE", "METRIC_COLUMN", "AVERAGE_VALUE", "CALCULATION_DATE"]
                            
                            detail_log_df = session.create_dataframe(detail_log_data, schema=detail_schema)
                            
                            final_detail_log_df = detail_log_df.with_column("EXECUTION_TIMESTAMP", current_timestamp())
                            
                            final_detail_log_df.write.mode("append").save_as_table("DQ_DETAILS_ROLLING_AVERAGE")
                            
                            log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value_logged, rule_type, logic_implementation, result_value, "PASS", None)
                            print(f"Successfully logged rolling average for '{rule_name}' on segment: {segment_value_logged}.")

                        except Exception as loop_err:
                            error_message = f"Failed to process and log result for a segment in '{rule_name}': {loop_err}"
                            logger.error(error_message, exc_info=True)
                            log_dq_result(session, dq_results_table, rule_id, rule_name, "PROCESSING_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)

            except (json.JSONDecodeError, ValueError) as e:
                # --- THIS IS THE FIX ---
                error_message = f"Configuration error for ROLLING_AVERAGE rule '{rule_name}': {e}"
                # --- END FIX ---
                logger.error(error_message)
                log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)
            except Exception as e:
                error_message = f"Error executing ROLLING_AVERAGE rule '{rule_name}': {e}"
                logger.error(error_message, exc_info=True)
                log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)

        else:
            collected_results = session.sql(logic_definition).collect()
            final_indicator, result_value, error_message = "PASS", 0, None
            if collected_results and collected_results[0][0] is not None:
                result_value = collected_results[0][0]
                final_indicator = "FAIL" if result_value > 0 else "PASS"
            elif collected_results:
                final_indicator, error_message = "FAIL", "SQL query returned a NULL result."
            else:
                final_indicator, error_message = "ERROR", "SQL query returned no results."
            log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, result_value if collected_results else None, final_indicator, error_message)

    except ValueError as e:
        error_message = f"Configuration or parsing error for SQL rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing SQL rule '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)

# --- NEW SNOWPARK DUPLICATE CHECK RULE ---
def execute_duplicate_check(session: Session, rule_row: Row, dq_results_table: str) -> None:
    """
    Executes a flexible duplicate check using Snowpark.
    - Identifies records where a specified set of columns are identical.
    - Supports case-insensitive comparisons for text columns.
    - Supports rounding for numeric columns before comparison.
    """
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_DUPLICATE_RULE')
    logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SNOWPARK_FUNC')
    rule_type = getattr(rule_row, 'RULE_TYPE', 'UNIQUENESS')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')
    logger.info(f"Executing Snowpark duplicate check rule: {rule_name}")

    try:
        # 1. Parse and Validate Parameters
        params = json.loads(parameters_str)
        source_table = params.get("source_table")
        details_table = params.get("details_table")
        columns_to_check = params.get("columns_to_check", [])
        case_insensitive_columns = params.get("case_insensitive_columns", [])
        numeric_columns_to_round = params.get("numeric_columns_to_round", {})

        if not all([source_table, details_table]) or not columns_to_check:
            raise ValueError("Missing required keys in PARAMETERS JSON (source_table, details_table, columns_to_check).")

        # 2. Build Grouping Expressions
        df = session.table(source_table)
        group_by_exprs = []
        for c in columns_to_check:
            if c in case_insensitive_columns:
                group_by_exprs.append(upper(col(c)).alias(c))
            elif c in numeric_columns_to_round:
                decimals = numeric_columns_to_round[c]
                group_by_exprs.append(snowpark_round(col(c), decimals).alias(c))
            else:
                group_by_exprs.append(col(c))

        # 3. Find and Count Duplicate Groups
        duplicates_df = (
            df.groupBy(*group_by_exprs)
            .agg(count(lit(1)).alias("DUPLICATE_COUNT"))
            .filter(col("DUPLICATE_COUNT") > 1)
            .cache_result() # Cache for performance since it's used twice
        )

        count_of_duplicate_groups = duplicates_df.count()

        # 4. Log Detailed Results if Duplicates are Found
        if count_of_duplicate_groups > 0:
            logger.info(f"Found {count_of_duplicate_groups} duplicate groups for rule '{rule_name}'. Logging details to '{details_table}'.")

            # Create the DUPLICATE_VALUES object
            object_args = []
            for c in columns_to_check:
                object_args.extend([lit(c), col(c)])

            details_to_log = duplicates_df.with_column("DUPLICATE_VALUES", object_construct(*object_args)) \
                                          .with_column("RULE_NAME", lit(rule_name))

            final_details_df = details_to_log.select(
                "RULE_NAME",
                "DUPLICATE_VALUES",
                "DUPLICATE_COUNT"
            ).with_column("EXECUTION_TIMESTAMP", current_timestamp())

            final_details_df.write.mode("append").save_as_table(details_table)

        # 5. Log Summary Result
        final_indicator = "FAIL" if count_of_duplicate_groups > 0 else "PASS"
        log_dq_result(
            session=session,
            dq_results_table=dq_results_table,
            rule_id=rule_id,
            rule_name=rule_name,
            segment_value='N/A', # As per rule spec
            rule_type=rule_type,
            logic_implementation=logic_implementation,
            result_value=count_of_duplicate_groups,
            indicator=final_indicator,
            error_message=None
        )

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Configuration error for duplicate check rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, 'CONFIG_ERROR', rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing Snowpark duplicate check for '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, 'EXECUTION_ERROR', rule_type, logic_implementation, "ERROR", "ERROR", error_message)
# --- END NEW RULE ---

# --- MISSING DATA & NULLS CHECK (SNOWPARK) [CORRECTED FUNCTION] ---
def execute_missing_data_nulls_check(session: Session, rule_row: Row, dq_results_table: str) -> None:
    """
    Executes a multi-faceted check for yesterday's data:
    1. Timeliness: Fails if no data exists for yesterday.
    2. Nulls: Fails if key columns contain NULLs.
    3. Missing Permutations: Fails if data permutations from T-2 are missing in T-1.
    4. Count Mismatch: Fails if the record count drops from T-2 to T-1.
    """
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_MISSING_DATA_RULE')
    logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SNOWPARK_FUNC')
    rule_type = getattr(rule_row, 'RULE_TYPE', 'MISSING_DATA_NULLS')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')
    logger.info(f"Executing Missing Data/Nulls check: {rule_name}")

    try:
        # 1. Parse and Validate Parameters
        params = json.loads(parameters_str)
        source_table = params.get("source_table")
        details_table = params.get("details_table")
        date_column = params.get("date_column")
        permutation_columns = params.get("permutation_columns", [])
        not_null_columns = params.get("not_null_columns", [])

        if not all([source_table, details_table, date_column, permutation_columns, not_null_columns]):
            raise ValueError("Missing required keys in PARAMETERS JSON (source_table, details_table, date_column, permutation_columns, not_null_columns).")

        # 2. Date Setup
        dates_df = session.sql("SELECT CURRENT_DATE() as today, DATEADD(day, -1, today) as yesterday, DATEADD(day, -2, today) as day_before").collect()
        yesterday_date = dates_df[0]['YESTERDAY']
        day_before_date = dates_df[0]['DAY_BEFORE']
        logger.info(f"Checking data for date: {yesterday_date}")
        
        # 3. Initialize
        failure_messages = []
        total_failures = 0
        df = session.table(source_table).with_column(date_column, to_date(col(date_column)))
        
        # --- CHECK 1: Timeliness ---
        yesterday_df = df.filter(col(date_column) == lit(yesterday_date)).cache_result()
        count_t1 = yesterday_df.count()

        if count_t1 == 0:
            msg = f"Timeliness Failure: No data found for {yesterday_date}."
            logger.warning(msg)
            failure_messages.append(msg)
            total_failures += 1
            
            df_to_log = session.create_dataframe(
                [(rule_name, 'TIMELINESS', yesterday_date, {'message': msg})],
                schema=['RULE_NAME', 'FAILURE_TYPE', 'FAILURE_DATE', 'DETAILS']
            )
            df_to_log.with_column("EXECUTION_TIMESTAMP", current_timestamp()).write.mode("append").save_as_table(details_table)
            
            log_dq_result(session, dq_results_table, rule_id, rule_name, "N/A", rule_type, logic_implementation, total_failures, "FAIL", "; ".join(failure_messages))
            return

        # --- CHECK 2: Null Violations ---
        null_check_condition = None
        for nc in not_null_columns:
            condition = col(nc).isNull()
            if null_check_condition is None:
                null_check_condition = condition
            else:
                null_check_condition = null_check_condition | condition
        
        null_rows_df = yesterday_df.filter(null_check_condition)
        null_count = null_rows_df.count()

        if null_count > 0:
            msg = f"Null Violation: Found {null_count} rows with nulls in key columns for {yesterday_date}."
            logger.warning(msg)
            failure_messages.append(msg)
            total_failures += null_count
            
            details_to_log = null_rows_df.select(
                lit(rule_name).alias("RULE_NAME"),
                lit("NULL_VIOLATION").alias("FAILURE_TYPE"),
                lit(yesterday_date).alias("FAILURE_DATE"),
                object_construct(*[val for item in permutation_columns for val in (lit(item), col(item))]).alias("DETAILS")
            )
            details_to_log.with_column("EXECUTION_TIMESTAMP", current_timestamp()).write.mode("append").save_as_table(details_table)

        # --- CHECK 3 & 4: Missing Permutations and Count Mismatch ---
        day_before_df = df.filter(col(date_column) == lit(day_before_date)).cache_result()
        count_t2 = day_before_df.count()

        # Check 3: Missing Permutations
        if count_t2 > 0:
            perms_t1 = yesterday_df.select(*permutation_columns).distinct()
            perms_t2 = day_before_df.select(*permutation_columns).distinct()
            missing_perms_df = perms_t2.subtract(perms_t1)
            missing_perms_count = missing_perms_df.count()

            if missing_perms_count > 0:
                msg = f"Missing Permutations: {missing_perms_count} permutations from {day_before_date} are missing on {yesterday_date}."
                logger.warning(msg)
                failure_messages.append(msg)
                total_failures += missing_perms_count
                
                details_to_log = missing_perms_df.select(
                    lit(rule_name).alias("RULE_NAME"),
                    lit("MISSING_PERMUTATION").alias("FAILURE_TYPE"),
                    lit(yesterday_date).alias("FAILURE_DATE"),
                    object_construct(*[val for item in permutation_columns for val in (lit(item), col(item))]).alias("DETAILS")
                )
                details_to_log.with_column("EXECUTION_TIMESTAMP", current_timestamp()).write.mode("append").save_as_table(details_table)

        # Check 4: Count Mismatch
        if count_t1 < count_t2:
            msg = f"Count Mismatch: Row count dropped from {count_t2} on {day_before_date} to {count_t1} on {yesterday_date}."
            logger.warning(msg)
            failure_messages.append(msg)
            total_failures += 1
            
            details_json = {'message': msg, 'yesterday_count': count_t1, 'previous_day_count': count_t2}
            
            df_to_log = session.create_dataframe(
                [(rule_name, 'COUNT_MISMATCH', yesterday_date, details_json)],
                schema=['RULE_NAME', 'FAILURE_TYPE', 'FAILURE_DATE', 'DETAILS']
            )
            df_to_log.with_column("EXECUTION_TIMESTAMP", current_timestamp()).write.mode("append").save_as_table(details_table)
            
        # 4. Final Logging
        final_indicator = "FAIL" if total_failures > 0 else "PASS"
        error_summary = "; ".join(failure_messages) if failure_messages else None
        
        log_dq_result(session, dq_results_table, rule_id, rule_name, "Daily Check", rule_type, logic_implementation, total_failures, final_indicator, error_summary)

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Configuration error for Missing Data/Nulls rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "CONFIG_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing Missing Data/Nulls rule '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "EXECUTION_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)


# --- SPIKE/DIP CHECK RULE EXECUTION (SNOWPARK) ---
def execute_spike_dip_check(session: Session, rule_row: Row, dq_results_table: str) -> None:
    """
    Executes a Spike or Dip check using Snowpark DataFrames.
    Identifies if the latest data point for a segment has changed by a given
    percentage threshold compared to the previous data point.
    It logs all historical spikes/dips to a details table for visualization.
    (Compatible with older Snowpark versions without .qualify())
    """
    # This import is needed for the row_number function
    from snowflake.snowpark.functions import row_number

    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_SPIKE_DIP_RULE')
    logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SNOWPARK_FUNC')
    rule_type = getattr(rule_row, 'RULE_TYPE', 'SPIKE_DIP_CHECK')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')

    logger.info(f"Executing Spike/Dip check rule: {rule_name}")

    try:
        # 1. Parse Parameters
        params = json.loads(parameters_str)
        source_table = params.get("source_table")
        details_table = params.get("details_table")
        metric_column = params.get("metric_column")
        timestamp_col = params.get("timestamp_col")
        segment_columns = params.get("segment_columns", [])
        threshold = float(params.get("threshold", 0.50))

        if not all([source_table, details_table, metric_column, timestamp_col]):
            raise ValueError("Missing required keys in PARAMETERS JSON (source_table, details_table, metric_column, timestamp_col).")
        if not segment_columns:
            raise ValueError("`segment_columns` must be a non-empty list in PARAMETERS JSON.")

        # 2. Snowpark DataFrame Logic
        df = session.table(source_table)
        window_spec = Window.partitionBy(*[col(c) for c in segment_columns]).orderBy(col(timestamp_col))

        df_with_prev_value = df.with_column("prev_metric", lag(col(metric_column), 1).over(window_spec))

        df_with_change = df_with_prev_value.with_column(
            "pct_change",
            when(col("prev_metric").isNotNull() & (col("prev_metric") != 0),
                (col(metric_column) - col("prev_metric")) / col("prev_metric")
            ).otherwise(None)
        )

        # 3. Identify all spikes/dips for the details table
        all_spikes_dips_df = df_with_change.filter(
            col("pct_change").isNotNull() & (snowpark_abs(col("pct_change")) >= threshold)
        )

        # 4. Log all detected spikes/dips to the details table
        if all_spikes_dips_df.count() > 0:
            logger.info(f"Found historical spikes/dips for rule '{rule_name}'. Logging to '{details_table}'.")
            
            object_args = []
            for p_col in segment_columns:
                object_args.append(lit(p_col))
                object_args.append(col(p_col))
            segment_values_obj = object_construct(*object_args)
            
            details_to_log = all_spikes_dips_df.select(
                lit(rule_name).alias("RULE_NAME"),
                col(timestamp_col).alias("EVENT_DATE"),
                segment_values_obj.alias("SEGMENT_VALUES"),
                col(metric_column).alias("METRIC_VALUE"),
                col("prev_metric").alias("PREVIOUS_METRIC_VALUE"),
                col("pct_change").alias("PERCENT_CHANGE")
            ).with_column("EXECUTION_TIMESTAMP", current_timestamp())

            details_to_log.write.mode("append").save_as_table(details_table)

        # --- REPLACEMENT FOR .qualify() ---
        # 5. Check only the LATEST data point for the main DQ_RESULTS log
        # Define a window to rank rows within each segment by date, descending
        ranking_window_spec = Window.partitionBy(*[col(c) for c in segment_columns]).orderBy(col(timestamp_col).desc())

        # Add a row number to each record; the latest record will have rn = 1
        df_with_rank = df_with_change.with_column("rn", row_number().over(ranking_window_spec))

        # Filter for only the latest record in each segment (rn = 1)
        latest_points_df = df_with_rank.filter(col("rn") == 1)
        # --- END REPLACEMENT ---
        
        # Filter to find segments where the latest point is a spike/dip
        latest_spikes = latest_points_df.filter(
            col("pct_change").isNotNull() & (snowpark_abs(col("pct_change")) >= threshold)
        ).collect()

        count_of_latest_spikes = len(latest_spikes)
        
        # 6. Log the main summary result
        result_value = count_of_latest_spikes
        final_indicator = "FAIL" if result_value > 0 else "PASS"
        error_message = f"{count_of_latest_spikes} segment(s) failed the spike/dip check on the latest data." if result_value > 0 else None
        
        segment_value_summary = json.dumps({"segments": segment_columns})
        
        log_dq_result(
            session=session,
            dq_results_table=dq_results_table,
            rule_id=rule_id,
            rule_name=rule_name,
            segment_value=segment_value_summary,
            rule_type=rule_type,
            logic_implementation=logic_implementation,
            result_value=result_value,
            indicator=final_indicator,
            error_message=error_message
        )

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Configuration error for spike/dip rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "CONFIG_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing Snowpark spike/dip rule '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "EXECUTION_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)


# --- ANOMALY DETECTION RULE EXECUTION (CORTEX) --- [FINAL CORRECTED FUNCTION]
def execute_detect_anomalies(session: Session, rule_row: Row, dq_results_table: str) -> None:
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_ANOMALY_RULE')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')
    segment_value = getattr(rule_row, 'SEGMENT_VALUE', None)
    logic_implementation, rule_type = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'CORTEX_DETECT_ANOMALIES'), getattr(rule_row, 'RULE_TYPE', 'ANOMALY_DETECTION')
    
    temp_view_name = ""
    try:
        params = json.loads(parameters_str)
        model_name = params.get("model_name")
        input_view = params.get("input_view")
        result_table = params.get("result_table")
        timestamp_col = params.get("timestamp_col")
        target_col = params.get("target_col")
        config_object = params.get("config_object", {})

        if not all([model_name, input_view, result_table, timestamp_col, target_col]):
            raise ValueError("Missing required keys in PARAMETERS JSON (model_name, input_view, result_table, timestamp_col, target_col).")

        # --- CACHE BUSTING LOGIC START ---
        # Create a unique temporary view for each rule execution to defeat Snowflake's result cache.
        # This forces the DETECT_ANOMALIES function to re-evaluate with the specific config_object.
        timestamp_suffix = int(time.time() * 1000)
        temp_view_name = f"TEMP_VIEW_{rule_id}_{timestamp_suffix}"
        
        create_temp_view_sql = f"""
        CREATE OR REPLACE TEMPORARY VIEW {temp_view_name} AS
        SELECT *, '{rule_id}' AS CACHE_BUSTER FROM {input_view}
        """
        session.sql(create_temp_view_sql).collect()
        logger.info(f"Created temporary cache-busting view '{temp_view_name}' for rule '{rule_name}'.")
        # --- CACHE BUSTING LOGIC END ---

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Invalid PARAMETERS JSON for rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)
        return

    logger.info(f"Executing anomaly detection rule: {rule_name}")
    try:
        # Use a Snowflake Scripting block to atomically run the call and create the table.
        # It now points to our unique temporary view.
        call_params = f"INPUT_DATA => SYSTEM$REFERENCE('VIEW', '{temp_view_name}'), TIMESTAMP_COLNAME => '{timestamp_col}', TARGET_COLNAME => '{target_col}'"
        if config_object:
            config_string = json.dumps(config_object)
            call_params += f", CONFIG_OBJECT => PARSE_JSON('{config_string}')"
            
        snowflake_scripting_block = f"""
        BEGIN
            CALL {model_name}!DETECT_ANOMALIES({call_params});
            CREATE OR REPLACE TABLE {result_table} AS SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
        END;
        """
        logger.info(f"Running Snowflake Scripting block for rule '{rule_name}' on view '{temp_view_name}'...")
        session.sql(snowflake_scripting_block).collect()
        logger.info(f"Successfully created result table '{result_table}' for rule '{rule_name}'.")

        # Use the Snowpark DataFrame API for validation.
        logger.info(f"Validating results in '{result_table}' using DataFrame API.")
        
        result_value = session.table(result_table).filter(col("IS_ANOMALY") == True).count()
        logger.info(f"Validation for rule '{rule_name}' found {result_value} anomalies.")

        final_indicator = "FAIL" if result_value > 0 else "PASS"
        
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, result_value, final_indicator, None)

    except Exception as e:
        error_message = f"UNEXPECTED ERROR during anomaly detection for '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)


# --- HISTORICAL COMPLETENESS CHECK RULE EXECUTION (SNOWPARK) ---
def execute_historical_completeness_check(session: Session, rule_row: Row, dq_results_table: str) -> None:
    """
    Executes a Historical Completeness check using Snowpark DataFrames.
    Monitors record count consistency over 60-day rolling windows by:
    1. Ingesting daily record counts for the most recent day and previous 60 days
    2. Calculating expected change by removing 61st day count and adding newest day count
    3. Comparing actual vs expected cumulative count for current 60-day window
    4. Flagging discrepancies outside tolerance threshold
    5. Logging detailed audit trail for investigation
    """
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_HISTORICAL_COMPLETENESS_RULE')
    logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SNOWPARK_FUNC')
    rule_type = getattr(rule_row, 'RULE_TYPE', 'HISTORICAL_COMPLETENESS')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')
    
    logger.info(f"Executing Historical Completeness check: {rule_name}")

    try:
        # 1. Parse and Validate Parameters
        params = json.loads(parameters_str)
        source_table = params.get("source_table")
        details_table = params.get("details_table")
        date_column = params.get("date_column")
        count_column = params.get("count_column")
        dataset_column = params.get("dataset_column")
        tolerance_threshold = float(params.get("tolerance_threshold", 0.05))
        lookback_days = int(params.get("lookback_days", 60))

        if not all([source_table, details_table, date_column, count_column, dataset_column]):
            raise ValueError("Missing required keys in PARAMETERS JSON (source_table, details_table, date_column, count_column, dataset_column).")

        # 2. Setup date ranges with fallback logic
        dates_df = session.sql(f"""
            SELECT 
                CURRENT_DATE() as today,
                DATEADD(day, -1, CURRENT_DATE()) as expected_newest_day,
                DATEADD(day, -{lookback_days + 1}, CURRENT_DATE()) as expected_dropped_day,
                DATEADD(day, -{lookback_days}, CURRENT_DATE()) as expected_window_start
        """).collect()
        
        expected_newest_day = dates_df[0]['EXPECTED_NEWEST_DAY']
        expected_dropped_day = dates_df[0]['EXPECTED_DROPPED_DAY']
        expected_window_start = dates_df[0]['EXPECTED_WINDOW_START']
        
        # Check if yesterday's data exists, if not use the maximum available date
        df = session.table(source_table).with_column(date_column, to_date(col(date_column)))
        
        # Debug: Let's see what data is actually in the table
        logger.info("=== DEBUGGING DATA AVAILABILITY ===")
        try:
            sample_data = df.limit(5).collect()
            logger.info(f"Sample data from {source_table}: {sample_data}")
            
            date_range = df.agg({date_column: "min", date_column: "max"}).collect()
            logger.info(f"Date range in {source_table}: {date_range}")
            
            unique_brands = df.select(dataset_column).distinct().collect()
            logger.info(f"Unique brands in {source_table}: {unique_brands}")
            
        except Exception as debug_error:
            logger.error(f"Error during debug data check: {debug_error}")
        logger.info("=== END DEBUGGING ===")
        
        # Check for yesterday's data
        yesterday_check = df.filter(col(date_column) == lit(expected_newest_day)).count()
        logger.info(f"Records found for expected date {expected_newest_day}: {yesterday_check}")
        
        data_lag_message = None
        if yesterday_check == 0:
            # No data for yesterday, find the maximum available date
            logger.info(f"No data found for expected date {expected_newest_day}, falling back to maximum available date")
            
            try:
                # Use a more robust way to get the max date
                max_date_query = f"SELECT MAX({date_column}) as max_date FROM {source_table}"
                max_date_result = session.sql(max_date_query).collect()
                
                if max_date_result and len(max_date_result) > 0:
                    actual_newest_day = max_date_result[0]['MAX_DATE']
                    logger.info(f"Found maximum date: {actual_newest_day}")
                else:
                    raise ValueError(f"Could not determine maximum date from {source_table}")
                
                if actual_newest_day is None:
                    raise ValueError(f"No data found in table {source_table}")
                
                # Calculate dates based on actual newest day using SQL for reliability
                date_calc_query = f"""
                SELECT 
                    '{actual_newest_day}'::DATE as newest_day,
                    DATEADD(day, -{lookback_days + 1}, '{actual_newest_day}'::DATE) as dropped_day,
                    DATEADD(day, -{lookback_days}, '{actual_newest_day}'::DATE) as window_start,
                    DATEDIFF(day, '{actual_newest_day}'::DATE, '{expected_newest_day}'::DATE) as days_behind
                """
                date_calc_result = session.sql(date_calc_query).collect()[0]
                
                newest_day = date_calc_result['NEWEST_DAY']
                dropped_day = date_calc_result['DROPPED_DAY']
                window_start = date_calc_result['WINDOW_START']
                days_behind = date_calc_result['DAYS_BEHIND']
                
                data_lag_message = f"Data is {days_behind} day(s) behind expected date. Using {actual_newest_day} instead of {expected_newest_day}."
                logger.warning(data_lag_message)
                
            except Exception as fallback_error:
                logger.error(f"Error in fallback date logic: {fallback_error}")
                raise ValueError(f"Could not determine valid dates for analysis: {fallback_error}")
        else:
            # Use expected dates (yesterday's data exists)
            newest_day = expected_newest_day
            dropped_day = expected_dropped_day
            window_start = expected_window_start
            logger.info(f"Data is up-to-date. Using expected date {expected_newest_day}")
        
        logger.info(f"Checking historical completeness for newest day: {newest_day}, window start: {window_start}, dropped day: {dropped_day}")

        # 3. Get the source data and prepare for analysis
        
        # 4. Get unique datasets to process
        datasets = df.select(dataset_column).distinct().collect()
        
        total_failed_datasets = 0
        import time
        execution_id = f"HIST_COMP_{rule_id}_{int(time.time() * 1000)}"
        
        for dataset_row in datasets:
            # Handle column name case-insensitively and provide fallback
            dataset_name = None
            for attr_name in dir(dataset_row):
                if attr_name.upper() == dataset_column.upper():
                    dataset_name = getattr(dataset_row, attr_name)
                    break
            
            # Fallback if column name doesn't match
            if dataset_name is None:
                dataset_name = getattr(dataset_row, dataset_column, None)
            
            # Final fallback if still None
            if dataset_name is None:
                logger.warning(f"Could not extract dataset name from row using column '{dataset_column}'. Skipping dataset.")
                continue
            
            try:
                # Filter data for current dataset
                dataset_df = df.filter(col(dataset_column) == lit(dataset_name))
                logger.info(f"Processing dataset: {dataset_name}")
                
                # Debug: Check if we have any data for this dataset
                total_records = dataset_df.count()
                logger.info(f"Total records for dataset {dataset_name}: {total_records}")
                
                # Get the newest day count - handle multiple rows by summing
                try:
                    newest_day_query = f"""
                    SELECT SUM({count_column}) as daily_total
                    FROM {source_table} 
                    WHERE {date_column} = '{newest_day}' 
                    AND {dataset_column} = '{dataset_name}'
                    """
                    newest_result = session.sql(newest_day_query).collect()
                    logger.info(f"Newest day query result for {dataset_name}: {newest_result}")
                    
                    if newest_result and len(newest_result) > 0 and newest_result[0]['DAILY_TOTAL'] is not None:
                        newest_record_count = newest_result[0]['DAILY_TOTAL']
                    else:
                        newest_record_count = 0
                        
                except Exception as newest_error:
                    logger.error(f"Error getting newest day count for {dataset_name}: {newest_error}")
                    newest_record_count = 0
                
                logger.info(f"Newest day record count for {dataset_name}: {newest_record_count}")
                
                # Get the dropped day count - handle multiple rows by summing
                try:
                    dropped_day_query = f"""
                    SELECT SUM({count_column}) as daily_total
                    FROM {source_table} 
                    WHERE {date_column} = '{dropped_day}' 
                    AND {dataset_column} = '{dataset_name}'
                    """
                    dropped_result = session.sql(dropped_day_query).collect()
                    logger.info(f"Dropped day query result for {dataset_name}: {dropped_result}")
                    
                    if dropped_result and len(dropped_result) > 0 and dropped_result[0]['DAILY_TOTAL'] is not None:
                        dropped_record_count = dropped_result[0]['DAILY_TOTAL']
                    else:
                        dropped_record_count = 0
                        
                except Exception as dropped_error:
                    logger.error(f"Error getting dropped day count for {dataset_name}: {dropped_error}")
                    dropped_record_count = 0
                
                logger.info(f"Dropped day record count for {dataset_name}: {dropped_record_count}")
                
                # Calculate previous cumulative count - simplified approach
                try:
                    previous_cumulative_query = f"""
                    SELECT SUM({count_column}) as total_count
                    FROM {source_table} 
                    WHERE {date_column} >= '{window_start}' 
                    AND {date_column} < '{newest_day}'
                    AND {dataset_column} = '{dataset_name}'
                    """
                    previous_cumulative_result = session.sql(previous_cumulative_query).collect()
                    logger.info(f"Previous cumulative query result for {dataset_name}: {previous_cumulative_result}")
                    
                    if previous_cumulative_result and len(previous_cumulative_result) > 0:
                        total_count_value = previous_cumulative_result[0]['TOTAL_COUNT']
                        previous_cumulative_count = total_count_value if total_count_value is not None else 0
                    else:
                        previous_cumulative_count = 0
                except Exception as prev_cumulative_error:
                    logger.error(f"Error calculating previous cumulative count for {dataset_name}: {prev_cumulative_error}")
                    previous_cumulative_count = 0
                
                logger.info(f"Previous cumulative count for {dataset_name}: {previous_cumulative_count}")
                
                # Calculate actual current 60-day window count - simplified approach
                try:
                    actual_60_day_query = f"""
                    SELECT SUM({count_column}) as total_count
                    FROM {source_table} 
                    WHERE {date_column} >= '{window_start}' 
                    AND {date_column} <= '{newest_day}'
                    AND {dataset_column} = '{dataset_name}'
                    """
                    actual_60_day_result = session.sql(actual_60_day_query).collect()
                    logger.info(f"Actual 60-day query result for {dataset_name}: {actual_60_day_result}")
                    
                    if actual_60_day_result and len(actual_60_day_result) > 0:
                        total_count_value = actual_60_day_result[0]['TOTAL_COUNT']
                        actual_60_day_count = total_count_value if total_count_value is not None else 0
                    else:
                        actual_60_day_count = 0
                except Exception as actual_60_error:
                    logger.error(f"Error calculating actual 60-day count for {dataset_name}: {actual_60_error}")
                    actual_60_day_count = 0
                
                logger.info(f"Actual 60-day count for {dataset_name}: {actual_60_day_count}")
                
                # REQUIREMENTS LOGIC: Rolling 60-day window consistency check
                # Expected = Previous_Cumulative_59_Days - Dropped_Day_60 + Newest_Day_60
                expected_60_day_count = previous_cumulative_count - dropped_record_count + newest_record_count
                
                logger.info(f"Expected 60-day count for {dataset_name}: {expected_60_day_count} (previous: {previous_cumulative_count} - dropped: {dropped_record_count} + newest: {newest_record_count})")
                
                # Calculate deviation between actual vs expected rolling window
                deviation = actual_60_day_count - expected_60_day_count
                
                # Handle division by zero case
                if expected_60_day_count != 0:
                    deviation_percentage = abs(deviation) / expected_60_day_count
                else:
                    deviation_percentage = 0.0
                    logger.warning(f"Expected count is zero for {dataset_name}, setting deviation percentage to 0")
                
                logger.info(f"Deviation for {dataset_name}: {deviation} ({deviation_percentage:.4f}%)")
                
                # Determine status
                status = "FAIL" if deviation_percentage > tolerance_threshold else "PASS"
                if status == "FAIL":
                    total_failed_datasets += 1
                
                logger.info(f"Status for {dataset_name}: {status}")
                
                # Log detailed results - following audit trail requirements per spec
                detail_data = {
                    "EXECUTION_ID": execution_id,
                    "RULE_ID": rule_id,
                    "DATASET_NAME": dataset_name,
                    "DATA_DATE": newest_day,
                    "ACTUAL_60_DAY_COUNT": actual_60_day_count,
                    "EXPECTED_60_DAY_COUNT": expected_60_day_count,
                    "DEVIATION": deviation,
                    "DEVIATION_PERCENTAGE": float(round(deviation_percentage, 4)),
                    "TOLERANCE_THRESHOLD": tolerance_threshold,
                    "STATUS": status,
                    "DROPPED_DAY_DATE": dropped_day,
                    "DROPPED_DAY_COUNT": dropped_record_count,
                    "NEWEST_DAY_COUNT": newest_record_count,
                    "PREVIOUS_CUMULATIVE_COUNT": previous_cumulative_count
                    # Comprehensive audit trail as per requirements
                }
                
                details_df = session.create_dataframe([detail_data], schema=list(detail_data.keys()))
                final_details_df = details_df.with_column("EXECUTION_TIMESTAMP", current_timestamp())
                final_details_df.write.mode("append").save_as_table(details_table)
                
                logger.info(f"Dataset '{dataset_name}': Status={status}, Actual={actual_60_day_count}, Expected={expected_60_day_count}, Deviation={deviation_percentage:.4f}")
                
            except Exception as dataset_error:
                logger.error(f"Error processing dataset '{dataset_name}' in Historical Completeness check: {dataset_error}")
                
                # Log error details
                error_detail_data = {
                    "EXECUTION_ID": execution_id,
                    "RULE_ID": rule_id,
                    "DATASET_NAME": dataset_name,
                    "DATA_DATE": newest_day,
                    "ACTUAL_60_DAY_COUNT": None,
                    "EXPECTED_60_DAY_COUNT": None,
                    "DEVIATION": None,
                    "DEVIATION_PERCENTAGE": None,
                    "TOLERANCE_THRESHOLD": tolerance_threshold,
                    "STATUS": "ERROR",
                    "DROPPED_DAY_DATE": dropped_day,
                    "DROPPED_DAY_COUNT": None,
                    "NEWEST_DAY_COUNT": None,
                    "PREVIOUS_CUMULATIVE_COUNT": None
                }
                
                error_details_df = session.create_dataframe([error_detail_data], schema=list(error_detail_data.keys()))
                error_final_details_df = error_details_df.with_column("EXECUTION_TIMESTAMP", current_timestamp())
                error_final_details_df.write.mode("append").save_as_table(details_table)
                
                total_failed_datasets += 1
        
        # 5. Log summary result to main DQ_RESULTS table
        final_indicator = "FAIL" if total_failed_datasets > 0 else "PASS"
        error_message = f"{total_failed_datasets} dataset(s) failed the historical completeness check." if total_failed_datasets > 0 else None
        
        log_dq_result(
            session=session,
            dq_results_table=dq_results_table,
            rule_id=rule_id,
            rule_name=rule_name,
            segment_value=f"60-day window analysis",
            rule_type=rule_type,
            logic_implementation=logic_implementation,
            result_value=total_failed_datasets,
            indicator=final_indicator,
            error_message=error_message
        )
        
        logger.info(f"Historical Completeness check completed. Failed datasets: {total_failed_datasets}")

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Configuration error for Historical Completeness rule '{rule_name}': {e}"
        logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "CONFIG_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing Historical Completeness rule '{rule_name}': {e}"
        logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, "EXECUTION_ERROR", rule_type, logic_implementation, "ERROR", "ERROR", error_message)


def execute_snowpark_func_rule(session: Session, rule_row: Row, dq_results_table: str) -> None:
    rule_id, rule_name = getattr(rule_row, 'RULE_ID', None), getattr(rule_row, 'RULE_NAME', 'UNKNOWN_SNOWPARK_RULE')
    logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SNOWPARK_FUNC')
    segment_value = getattr(rule_row, 'SEGMENT_VALUE', None)
    rule_type = getattr(rule_row, 'RULE_TYPE', 'CUSTOM_SNOWPARK')
    parameters_str = getattr(rule_row, 'PARAMETERS', '{}')
    metric_col = getattr(rule_row, 'DATA_SOURCE_METRIC', None)
    segment_column_str = getattr(rule_row, 'SEGMENT_COLUMN', None)

    logger.info(f"Executing embedded sustained trend analysis for rule: {rule_name}")
    try:
        params = json.loads(parameters_str)
        source_table, details_table, timestamp_col, trend_threshold = params.get("source_table"), params.get("details_table"), params.get("timestamp_col"), params.get("trend_days_threshold")
        if not all([source_table, details_table, timestamp_col, trend_threshold]):
            raise ValueError("Missing required keys in PARAMETERS JSON (source_table, details_table, timestamp_col, trend_days_threshold).")
        if not metric_col or not segment_column_str:
            raise ValueError("Missing required columns from RULE_CATALOG (DATA_SOURCE_METRIC, SEGMENT_COLUMN).")

        partition_cols = [c.strip() for c in segment_column_str.split(',') if c.strip()]
        if not partition_cols:
            raise ValueError("SEGMENT_COLUMN is empty or invalid.")

        df = session.table(source_table)
        window_spec = Window.partitionBy(*[col(p) for p in partition_cols]).orderBy(timestamp_col)
        df_with_prev_value = df.with_column("prev_metric", lag(col(metric_col), 1).over(window_spec))
        df_with_groups = df_with_prev_value.with_column("is_increasing", when(col(metric_col) > col("prev_metric"), 1).otherwise(0)).with_column("trend_group_start", when(col("is_increasing") == 0, 1).otherwise(0)).with_column("trend_group_id", snowpark_sum("trend_group_start").over(window_spec))
        trend_lengths_df = (df_with_groups.filter(col("is_increasing") == 1).groupBy(*partition_cols, "trend_group_id").count().withColumnRenamed("COUNT", "trend_length"))
        sustained_trends = trend_lengths_df.filter(col("trend_length") >= lit(trend_threshold))
        count_of_trends = sustained_trends.count()
        
        if count_of_trends > 0:
            logger.info(f"Found {count_of_trends} sustained trends. Saving details to '{details_table}'.")
            object_args = []
            for p_col in partition_cols:
                object_args.append(lit(p_col))
                object_args.append(col(p_col))
            segment_values_obj = object_construct(*object_args)
            details_to_log = sustained_trends.with_column("RULE_NAME", lit(rule_name)).with_column("SEGMENT_COLUMNS", lit(", ".join(partition_cols))).with_column("SEGMENT_VALUES", segment_values_obj)
            final_details_df = details_to_log.select("RULE_NAME", "SEGMENT_COLUMNS", "SEGMENT_VALUES", col("trend_length").alias("TREND_LENGTH")).with_column("EXECUTION_TIMESTAMP", current_timestamp())
            final_details_df.write.mode("append").save_as_table(details_table)
        
        result_value = count_of_trends
        final_indicator = "FAIL" if result_value > 0 else "PASS"
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, result_value, final_indicator, None)

    except ValueError as e:
        error_message = f"Configuration error for sustained trend rule '{rule_name}': {e}"; logger.error(error_message)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)
    except Exception as e:
        error_message = f"Error executing embedded Snowpark function for rule '{rule_name}': {e}"; logger.error(error_message, exc_info=True)
        log_dq_result(session, dq_results_table, rule_id, rule_name, segment_value, rule_type, logic_implementation, "ERROR", "ERROR", error_message)