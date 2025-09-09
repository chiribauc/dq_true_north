# dq_engine.py
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session, Row
from snowflake.snowpark.functions import col, current_date, current_timestamp
from snowflake.snowpark.exceptions import SnowparkSQLException
import json
import logging

# Import the library functions
import dq_rules_library as dqlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MAIN ORCHESTRATION FUNCTION ---
def main(session: Session):
    """
    Orchestrates the execution of data quality rules by reading from RULE_CATALOG
    and dispatching to appropriate rule execution functions.
    """
    print("Starting data quality rule execution...")
    dq_results_table_name = "DQ_RESULTS"
    try:
        rule_catalog_rows = session.table("RULE_CATALOG").filter(col("IS_ACTIVE") == True).collect()
        if not rule_catalog_rows:
            print("No active rules found in the catalog.")
            return session.create_dataframe([], schema=["RULE_ID", "RULE_NAME", "SEGMENT_VALUE", "RULE_TYPE", "LOGIC_IMPLEMENTATION", "EXECUTION_TIMESTAMP", "RESULT_VALUE", "INDICATOR", "ERROR_MESSAGE"])
        print(f"Found {len(rule_catalog_rows)} active rules to process.")

        try:
            session.sql(f"SELECT 1 FROM {dq_results_table_name} LIMIT 1").collect()
        except SnowparkSQLException:
            print(f"DQ_RESULTS table '{dq_results_table_name}' does not exist. Creating it.")
            create_table_sql = f"""
            CREATE OR REPLACE TABLE {dq_results_table_name} (
            RULE_ID INT, RULE_NAME VARCHAR(255) NOT NULL, SEGMENT_VALUE VARCHAR(255),
            RULE_TYPE VARCHAR(50), LOGIC_IMPLEMENTATION VARCHAR(100),
            EXECUTION_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            RESULT_VALUE VARCHAR(1000), INDICATOR VARCHAR(20), ERROR_MESSAGE VARCHAR(4000)
            )"""
            session.sql(create_table_sql).collect()
            print(f"Table '{dq_results_table_name}' created.")

        for rule_row in rule_catalog_rows:
            try:
                rule_type = getattr(rule_row, 'RULE_TYPE', None)
                logic_implementation = getattr(rule_row, 'LOGIC_IMPLEMENTATION', None)
                
                # Dispatch based on rule type and implementation
                if rule_type == 'UNIQUENESS' and logic_implementation == 'SQL_QUERY':
                    dqlib.execute_sql_rule(session, rule_row, dq_results_table_name)
                elif rule_type == 'COMPLETENESS' and logic_implementation == 'SQL_QUERY':
                    dqlib.execute_sql_rule(session, rule_row, dq_results_table_name)
                elif rule_type == 'ROLLING_AVERAGE' and logic_implementation == 'SQL_QUERY':
                    dqlib.execute_sql_rule(session, rule_row, dq_results_table_name)
                elif logic_implementation == 'SQL_QUERY':
                    dqlib.execute_sql_rule(session, rule_row, dq_results_table_name)
                # NEW DISPATCH: SPIKE_DIP_CHECK rule
                elif rule_type == 'SPIKE_DIP_CHECK' and logic_implementation == 'SNOWPARK_FUNC':
                    dqlib.execute_spike_dip_check(session, rule_row, dq_results_table_name)
                # NEW DISPATCH: MISSING_DATA_NULLS rule
                elif rule_type == 'MISSING_DATA_NULLS' and logic_implementation == 'SNOWPARK_FUNC':
                    dqlib.execute_missing_data_nulls_check(session, rule_row, dq_results_table_name)
                # NEW DISPATCH: HISTORICAL_COMPLETENESS rule
                elif rule_type == 'HISTORICAL_COMPLETENESS' and logic_implementation == 'SNOWPARK_FUNC':
                    dqlib.execute_historical_completeness_check(session, rule_row, dq_results_table_name)
                # NEW DISPATCH: UNIQUENESS rule with SNOWPARK_FUNC (for duplicate checks)
                elif rule_type == 'UNIQUENESS' and logic_implementation == 'SNOWPARK_FUNC':
                    dqlib.execute_duplicate_check(session, rule_row, dq_results_table_name)
                elif logic_implementation == 'CORTEX_DETECT_ANOMALIES':
                    dqlib.execute_detect_anomalies(session, rule_row, dq_results_table_name)
                elif logic_implementation == 'SNOWPARK_FUNC':
                    dqlib.execute_snowpark_func_rule(session, rule_row, dq_results_table_name)
                elif logic_implementation == 'CORTEX_LLM':
                    logger.warning(f"Skipping LLM rule '{getattr(rule_row, 'RULE_NAME', 'N/A')}' - implementation pending.")
                else:
                    if logic_implementation:
                        logger.warning(f"Unsupported LOGIC_IMPLEMENTATION: {logic_implementation} for rule {getattr(rule_row, 'RULE_NAME', 'N/A')}")
            except Exception as e:
                logger.error(f"Critical error processing rule {getattr(rule_row, 'RULE_NAME', 'UNKNOWN')}: ", exc_info=True)
                dqlib.log_dq_result(
                    session=session,
                    dq_results_table=dq_results_table_name,
                    rule_id=getattr(rule_row, 'RULE_ID', None),
                    rule_name=getattr(rule_row, 'RULE_NAME', 'CRITICAL_RULE_ERROR'),
                    segment_value=getattr(rule_row, 'SEGMENT_VALUE', 'N/A'),
                    rule_type=getattr(rule_row, 'RULE_TYPE', 'SYSTEM_ERROR'),
                    logic_implementation=getattr(rule_row, 'LOGIC_IMPLEMENTATION', 'SYSTEM_ERROR'),
                    result_value="ERROR",
                    indicator="ERROR",
                    error_message=f"Critical error during rule processing: {e}"
                )

        print("Data quality rule execution process completed.")
        return session.table(dq_results_table_name)
    except Exception as e:
        error_message = f"MAIN EXECUTION ERROR: {str(e)}"
        print(error_message)
        if session:
            dqlib.log_dq_result(
                session=session,
                dq_results_table=dq_results_table_name,
                rule_id=None,
                rule_name="Main_Orchestration",
                segment_value="N/A",
                rule_type="SYSTEM",
                logic_implementation="SYSTEM_PROCESS",
                result_value="ERROR",
                indicator="ERROR",
                error_message=error_message
            )
        return session.create_dataframe([], schema=["RULE_ID", "RULE_NAME", "SEGMENT_VALUE", "RULE_TYPE", "LOGIC_IMPLEMENTATION", "EXECUTION_TIMESTAMP", "RESULT_VALUE", "INDICATOR", "ERROR_MESSAGE"])
    finally:
        print("Worksheet execution finished.")