import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, date_trunc, current_timestamp, lit, avg, stddev, when
from snowflake.snowpark.window import Window
import json
import altair as alt

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="True North DQ Reports",
    page_icon="🤖",
    layout="wide",
)

# --- 2. Establish Snowflake Session & Helper Functions ---
try:
    session = get_active_session()
except Exception as e:
    st.error(f"Could not connect to Snowflake. Please run this app within a Snowsight worksheet or ensure your connection is configured. Error: {e}")
    st.stop()

def filter_df_by_brand(df: pd.DataFrame, brand_data_value: str) -> pd.DataFrame:
    """
    A robust function to filter a dataframe for a specific brand or for uncategorized errors.
    """
    json_col_candidates = ['DETAILS', 'DUPLICATE_VALUES', 'SEGMENT_VALUES', 'SEGMENT_VALUE']
    json_col_name = next((col for col in json_col_candidates if col in df.columns), None)

    # --- NEW LOGIC FOR UNCATEGORIZED FAILURES ---
    if brand_data_value == "__GLOBAL__":
        if not json_col_name:
            return pd.DataFrame(columns=df.columns) # Return empty if no details column exists

        def is_json_uncategorized(json_str):
            try:
                if isinstance(json_str, str):
                    if not json_str.strip() or json_str == '{}': return True
                    data = json.loads(json_str)
                elif isinstance(json_str, dict):
                    if not data: return True
                    data = json_str
                else: # Not a parsable type
                    return True
                # It's uncategorized if the BRAND key is missing or the value is null/empty
                return "BRAND" not in data or pd.isna(data.get("BRAND"))
            except (json.JSONDecodeError, TypeError, AttributeError):
                return True # Treat parse errors as uncategorized

        return df[df[json_col_name].apply(is_json_uncategorized)].copy()

    # --- EXISTING LOGIC FOR BRAND-SPECIFIC FILTERING ---
    name_filter = df['RULE_NAME'].str.contains(brand_data_value, case=False, na=False)
    global_summary_filter = pd.Series([False] * len(df), index=df.index)
    if 'RULE_TYPE' in df.columns:
        # Show UNIQUENESS rules - try multiple approaches to find them
        if brand_data_value == "__GLOBAL__":
            # For global view, show all UNIQUENESS rules
            global_summary_filter = (df['RULE_TYPE'] == 'UNIQUENESS')
        else:
            # For brand-specific view, show UNIQUENESS rules that might be related to this brand
            uniqueness_rules = (df['RULE_TYPE'] == 'UNIQUENESS')
            # Try different ways to match brand: rule name contains brand, or show all UNIQUENESS rules
            brand_in_name = df['RULE_NAME'].str.contains(brand_data_value, case=False, na=False)
            # For now, show ALL uniqueness rules on brand pages to debug
            global_summary_filter = uniqueness_rules

    json_filter = pd.Series([False] * len(df), index=df.index)
    if json_col_name:
        def check_brand_in_json(json_str):
            try:
                if isinstance(json_str, str):
                    if not json_str.strip(): return False
                    data = json.loads(json_str)
                elif isinstance(json_str, dict):
                    data = json_str
                else:
                    return False
                return data.get("BRAND") == brand_data_value
            except (json.JSONDecodeError, TypeError, AttributeError):
                return False
        json_filter = df[json_col_name].apply(check_brand_in_json)

    final_filter = global_summary_filter | name_filter | json_filter
    return df[final_filter].copy()

# --- AI Helper Functions (MODIFIED FOR CONTEXTUAL ANALYSIS) ---
RULE_DEFINITIONS = """
- **Uniqueness / Duplicate Check**: Verifies that records are unique. This can be a simple SQL-based check for duplicates on a given day or a more advanced check that finds records where multiple columns are identical, with special handling for case-insensitivity ('WSJ' = 'wsj') and numeric rounding.
- **Sustained Trend**: Monitors a metric for consecutive increases over time.
- **Anomaly Detection**: Uses machine learning to find unexpected spikes or drops.
- **Completeness Check**: Validates that yesterday's data is present.
- **Rolling Average**: Calculates the average of a metric over a sliding window of days to smooth out trends.
- **Spike or Dip**: Identifies individual data points that are significantly higher or lower than the previous day.
- **Missing Data/Nulls Check**: A comprehensive check for timeliness (data arrived), integrity (no nulls), completeness (no missing segments), and volume (no unexpected drops).
"""

def get_summary_prompt(data_as_json: str, brand: str, analysis_type: str) -> str:
    """Creates a more specific prompt for the LLM."""
    return f"You are a data quality analyst. Summarize the key findings from the following data, which represents '{analysis_type}' for the publication '{brand}'. Focus on the scale of the issue and what the numbers mean. Be concise and clear.\n\nData:\n{data_as_json}"

def get_help_prompt(user_question: str) -> str:
    return f"Based on this documentation:\n\n{RULE_DEFINITIONS}\n\nAnswer the user's question: '{user_question}'"

def call_cortex_llm(prompt: str) -> str:
    formatted_prompt = prompt.replace("'", "''")
    try:
        cortex_response_df = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3-8b', '{formatted_prompt}') as response").to_pandas()
        return cortex_response_df['RESPONSE'][0].strip()
    except Exception as e:
        return f"Error calling Cortex LLM: {e}"

def generate_ai_summary_for_tab(data_df: pd.DataFrame, analysis_type: str, brand_name: str, relevant_cols: list):
    """Creates a contextual AI summary section for a given DataFrame."""
    st.divider()
    st.subheader("Tab-Specific AI Analysis")
    if st.button(f"Generate AI Summary for these {analysis_type.lower()}", key=f"ai_{analysis_type.replace(' ', '_')}"):
        with st.spinner(f"AI is analyzing the {analysis_type.lower()}..."):
            if data_df is not None and not data_df.empty:
                data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                summary_prompt = get_summary_prompt(data_for_prompt, brand_name, analysis_type)
                summary = call_cortex_llm(summary_prompt)
                st.markdown(summary)
            else:
                st.success(f"No {analysis_type.lower()} were found to analyze for {brand_name}.")

def generate_ai_summary_for_duplicates(data_df: pd.DataFrame, brand_name: str):
    """Creates an optimized AI summary section specifically for duplicate records."""
    st.divider()
    st.subheader("Tab-Specific AI Analysis")
    if st.button("Generate AI Summary for these duplicate records", key="ai_duplicate_records"):
        with st.spinner("AI is analyzing the duplicate records..."):
            if data_df is not None and not data_df.empty:
                total_records = len(data_df)
                
                if total_records > 100:
                    # For large datasets, provide summary statistics instead of individual records
                    try:
                        # Calculate summary statistics
                        total_duplicate_count = data_df['DUPLICATE_COUNT'].sum() if 'DUPLICATE_COUNT' in data_df.columns else total_records
                        avg_duplicates = data_df['DUPLICATE_COUNT'].mean() if 'DUPLICATE_COUNT' in data_df.columns else 1
                        max_duplicates = data_df['DUPLICATE_COUNT'].max() if 'DUPLICATE_COUNT' in data_df.columns else 1
                        min_duplicates = data_df['DUPLICATE_COUNT'].min() if 'DUPLICATE_COUNT' in data_df.columns else 1
                        
                        # Get unique dates if available
                        unique_dates = []
                        date_columns = [col for col in data_df.columns if 'DATE' in col.upper()]
                        if date_columns:
                            unique_dates = data_df[date_columns[0]].nunique()
                        
                        # Create summary data for the prompt
                        summary_data = {
                            "total_duplicate_entries": total_records,
                            "total_affected_records": int(total_duplicate_count),
                            "average_duplicates_per_entry": round(avg_duplicates, 2),
                            "maximum_duplicates_for_single_entry": int(max_duplicates),
                            "minimum_duplicates_for_single_entry": int(min_duplicates),
                            "unique_dates_affected": unique_dates if unique_dates else "Not available"
                        }
                        
                        summary_prompt = f"""You are a data quality analyst. Analyze the following duplicate records summary for the publication '{brand_name}'. 
                        
                        IMPORTANT: This is a summary of {total_records} duplicate record entries (too many to analyze individually).
                        
                        Summary Statistics:
                        - Total duplicate entries found: {summary_data['total_duplicate_entries']}
                        - Total affected records: {summary_data['total_affected_records']}
                        - Average duplicates per entry: {summary_data['average_duplicates_per_entry']}
                        - Maximum duplicates for a single entry: {summary_data['maximum_duplicates_for_single_entry']}
                        - Minimum duplicates for a single entry: {summary_data['minimum_duplicates_for_single_entry']}
                        - Unique dates affected: {summary_data['unique_dates_affected']}
                        
                        Please provide insights on:
                        1. The scale and severity of the duplication issue
                        2. What these numbers indicate about data quality
                        3. Potential business impact
                        4. Recommended next steps for investigation
                        
                        Be concise and focus on actionable insights."""
                        
                    except Exception as e:
                        summary_prompt = f"""You are a data quality analyst. There are {total_records} duplicate record entries for '{brand_name}'. 
                        This is a large dataset that requires immediate attention. Please provide insights on:
                        1. The significance of having {total_records} duplicate entries
                        2. Potential business impact of this scale of duplication
                        3. Recommended immediate actions"""
                else:
                    # For smaller datasets, analyze individual records as before
                    relevant_cols = ['DUPLICATE_VALUES', 'DUPLICATE_COUNT'] if 'DUPLICATE_COUNT' in data_df.columns else list(data_df.columns)
                    # Only include columns that exist in the dataframe
                    relevant_cols = [col for col in relevant_cols if col in data_df.columns]
                    data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                    summary_prompt = get_summary_prompt(data_for_prompt, brand_name, "Duplicate Records")
                
                summary = call_cortex_llm(summary_prompt)
                st.markdown(summary)
            else:
                st.success(f"No duplicate records were found to analyze for {brand_name}.")

def generate_ai_summary_for_spike_dip(data_df: pd.DataFrame, brand_name: str):
    """Creates an optimized AI summary section specifically for spike and dip events."""
    st.divider()
    st.subheader("Tab-Specific AI Analysis")
    if st.button("Generate AI Summary for these spike and dip events", key="ai_spike_dip_events"):
        with st.spinner("AI is analyzing the spike and dip events..."):
            if data_df is not None and not data_df.empty:
                total_events = len(data_df)
                
                if total_events > 100:
                    # For large datasets, provide summary statistics instead of individual events
                    try:
                        # Calculate summary statistics
                        spike_count = len(data_df[data_df['PERCENT_CHANGE'] > 0]) if 'PERCENT_CHANGE' in data_df.columns else 0
                        dip_count = len(data_df[data_df['PERCENT_CHANGE'] < 0]) if 'PERCENT_CHANGE' in data_df.columns else 0
                        
                        if 'PERCENT_CHANGE' in data_df.columns:
                            avg_change = data_df['PERCENT_CHANGE'].mean()
                            max_spike = data_df['PERCENT_CHANGE'].max()
                            max_dip = data_df['PERCENT_CHANGE'].min()
                            std_change = data_df['PERCENT_CHANGE'].std()
                        else:
                            avg_change = max_spike = max_dip = std_change = "Not available"
                        
                        # Get unique dates if available
                        unique_dates = "Not available"
                        date_columns = [col for col in data_df.columns if 'DATE' in col.upper()]
                        if date_columns:
                            unique_dates = data_df[date_columns[0]].nunique()
                        
                        # Get unique segments if available
                        unique_segments = "Not available"
                        if 'SEGMENT_VALUES' in data_df.columns:
                            unique_segments = data_df['SEGMENT_VALUES'].nunique()
                        
                        # Create summary data for the prompt
                        summary_data = {
                            "total_events": total_events,
                            "spike_count": spike_count,
                            "dip_count": dip_count,
                            "average_percent_change": round(avg_change, 2) if isinstance(avg_change, (int, float)) else avg_change,
                            "maximum_spike_percent": round(max_spike, 2) if isinstance(max_spike, (int, float)) else max_spike,
                            "maximum_dip_percent": round(max_dip, 2) if isinstance(max_dip, (int, float)) else max_dip,
                            "change_volatility": round(std_change, 2) if isinstance(std_change, (int, float)) else std_change,
                            "unique_dates_affected": unique_dates,
                            "unique_segments_affected": unique_segments
                        }
                        
                        summary_prompt = f"""You are a data quality analyst. Analyze the following spike and dip events summary for the publication '{brand_name}'. 
                        
                        IMPORTANT: This is a summary of {total_events} spike and dip events (too many to analyze individually).
                        
                        Summary Statistics:
                        - Total spike/dip events: {summary_data['total_events']}
                        - Number of spikes (positive changes): {summary_data['spike_count']}
                        - Number of dips (negative changes): {summary_data['dip_count']}
                        - Average percent change: {summary_data['average_percent_change']}%
                        - Maximum spike: {summary_data['maximum_spike_percent']}%
                        - Maximum dip: {summary_data['maximum_dip_percent']}%
                        - Change volatility (std dev): {summary_data['change_volatility']}%
                        - Unique dates affected: {summary_data['unique_dates_affected']}
                        - Unique segments affected: {summary_data['unique_segments_affected']}
                        
                        Please provide insights on:
                        1. The frequency and severity of order volume fluctuations
                        2. Whether the data shows more spikes or dips and what this indicates
                        3. The volatility level and its business implications
                        4. Potential operational or market factors that could cause these patterns
                        5. Recommended monitoring and response strategies
                        
                        Be concise and focus on actionable business insights."""
                        
                    except Exception as e:
                        summary_prompt = f"""You are a data quality analyst. There are {total_events} spike and dip events for '{brand_name}'. 
                        This indicates significant volatility in order patterns that requires immediate attention. Please provide insights on:
                        1. The significance of having {total_events} volatility events
                        2. Potential business impact of this level of order fluctuation
                        3. Recommended immediate analysis and monitoring actions"""
                else:
                    # For smaller datasets, analyze individual events as before
                    relevant_cols = ['EVENT_DATE', 'METRIC_VALUE', 'PREVIOUS_METRIC_VALUE', 'PERCENT_CHANGE']
                    # Only include columns that exist in the dataframe
                    relevant_cols = [col for col in relevant_cols if col in data_df.columns]
                    data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                    summary_prompt = get_summary_prompt(data_for_prompt, brand_name, "Spike and Dip Events")
                
                summary = call_cortex_llm(summary_prompt)
                st.markdown(summary)
            else:
                st.success(f"No spike and dip events were found to analyze for {brand_name}.")

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.title("Views")
    st.markdown("---")
    if st.button("Global / Uncategorized", key="global_errors"):
        st.session_state['display_brand'] = "Global / Uncategorized"
        st.session_state['data_brand'] = "__GLOBAL__" # Special internal value for filtering
        st.session_state['selected_metric'] = "All Metrics"
    st.markdown("---")

    st.title("Publications")
    publications = {"WSJ": "WSJ", "BARRON'S": "Barrons", "MarketWatch": "MarketWatch"}
    for display_name, data_value in publications.items():
        st.markdown(f"**{display_name.upper()}**")
        if st.button("Orders", key=f"orders_{data_value}"):
            st.session_state['display_brand'] = display_name
            st.session_state['data_brand'] = data_value
            st.session_state['selected_metric'] = "Orders"

# --- 4. Main Page Layout ---
st.title("🤖 True North DQ Reports")
if 'display_brand' not in st.session_state:
    st.info("Please select a view from the sidebar to begin.")
    st.stop()

display_brand = st.session_state.display_brand
data_brand = st.session_state.data_brand
selected_metric = st.session_state.selected_metric
st.header(f"{display_brand} - {selected_metric}")

# --- 5. Main Dashboard Tabs ---
# *** Updated to reflect all rule types from the library ***
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Summary", 
    "Same Date Uniqueness Details", 
    "Completeness Details",
    "Sustained Trend Details", 
    "Missing Data Details", 
    "Spike / Dip Details", 
    "Sigma & Anomaly Analysis"
])

# --- TAB 1: Summary ---
with tab1:
    with st.spinner(f"Loading summary data for {display_brand}..."):
        try:
            all_summary_df = session.table("DQ_RESULTS").to_pandas()
            st.subheader("Filters")
            filter_cols = st.columns(2)
            with filter_cols[0]:
                indicator_options = ["PASS", "FAIL", "ERROR"]
                selected_indicators = st.multiselect("Filter by Status:", options=indicator_options, default=indicator_options, key="summary_status_filter")
            with filter_cols[1]:
                all_rule_types = all_summary_df['RULE_TYPE'].unique().tolist()
                all_option = "All Rule Types"
                rule_type_options = [all_option] + all_rule_types
                user_selected_types = st.multiselect("Filter by Rule Type:", options=rule_type_options, default=[all_option], key="summary_type_filter")
                selected_rule_types = all_rule_types if all_option in user_selected_types else user_selected_types

            brand_summary_df = filter_df_by_brand(all_summary_df, data_brand)
            filtered_summary_df = brand_summary_df[
                (brand_summary_df['INDICATOR'].isin(selected_indicators)) &
                (brand_summary_df['RULE_TYPE'].isin(selected_rule_types))
            ]
            st.divider()
            st.subheader("Key Metrics (for filtered data)")
            if not filtered_summary_df.empty:
                total_runs = len(filtered_summary_df)
                fail_count = len(filtered_summary_df[filtered_summary_df['INDICATOR'] == 'FAIL'])
                error_count = len(filtered_summary_df[filtered_summary_df['INDICATOR'] == 'ERROR'])
                pass_rate = (total_runs - fail_count - error_count) / total_runs * 100 if total_runs > 0 else 100
                
                # Add UNIQUENESS rule metrics
                uniqueness_count = len(filtered_summary_df[filtered_summary_df['RULE_TYPE'] == 'UNIQUENESS'])
                uniqueness_failures = len(filtered_summary_df[
                    (filtered_summary_df['RULE_TYPE'] == 'UNIQUENESS') & 
                    (filtered_summary_df['INDICATOR'] == 'FAIL')
                ])
                
                metric_cols = st.columns(4)
                metric_cols[0].metric("Total Rules Executed", f"{total_runs}")
                metric_cols[1].metric("Pass Rate", f"{pass_rate:.2f}%")
                metric_cols[2].metric("Total Failures/Errors", f"{fail_count + error_count}", delta=f"{fail_count + error_count} issues", delta_color="inverse")
                metric_cols[3].metric("UNIQUENESS Rules", f"{uniqueness_count}", help=f"Failed: {uniqueness_failures}")
                
                # Debug information
                if st.checkbox("Show Debug Info", key="debug_summary"):
                    st.write("**Debug Information:**")
                    st.write(f"Total rules in system: {len(all_summary_df)}")
                    st.write(f"Rules after brand filtering: {len(brand_summary_df)}")
                    st.write(f"Rules after status/type filtering: {len(filtered_summary_df)}")
                    
                    # Show all rule types available
                    all_rule_types = all_summary_df['RULE_TYPE'].value_counts()
                    st.write("**All Rule Types in System:**")
                    st.write(all_rule_types)
                    
                    # Show UNIQUENESS rules specifically
                    uniqueness_in_system = all_summary_df[all_summary_df['RULE_TYPE'] == 'UNIQUENESS']
                    if not uniqueness_in_system.empty:
                        st.write("**All UNIQUENESS Rules in System:**")
                        st.dataframe(uniqueness_in_system[['RULE_NAME', 'RULE_TYPE', 'INDICATOR', 'SEGMENT_VALUE']])
                    else:
                        st.write("**No UNIQUENESS rules found in the entire system!**")
                
                st.divider()
                st.subheader("Detailed Results (filtered)")
                st.dataframe(filtered_summary_df)
            else:
                st.info("No data matches the current filter selection.")
                # Show debug info even when no filtered data
                if st.checkbox("Show Debug Info (No Data)", key="debug_no_data"):
                    st.write("**Debug Information:**")
                    st.write(f"Total rules in system: {len(all_summary_df)}")
                    st.write(f"Rules after brand filtering: {len(brand_summary_df)}")
                    
                    # Show what's available
                    if len(brand_summary_df) > 0:
                        st.write("**Rules found for this brand:**")
                        st.dataframe(brand_summary_df[['RULE_NAME', 'RULE_TYPE', 'INDICATOR']])
                    else:
                        st.write("**No rules found for this brand after filtering**")
                        st.write("**All available rules in system:**")
                        st.dataframe(all_summary_df[['RULE_NAME', 'RULE_TYPE', 'INDICATOR', 'SEGMENT_VALUE']].head(10))
                
            generate_ai_summary_for_tab(
                data_df=filtered_summary_df[filtered_summary_df['INDICATOR'] == 'FAIL'],
                analysis_type="Overall Failures",
                brand_name=display_brand,
                relevant_cols=['RULE_NAME', 'RULE_TYPE', 'RESULT_VALUE']
            )
        except Exception as e:
            st.error(f"Could not load summary data: {e}")

# --- TAB 2: Same Date Uniqueness Details ---
with tab2:
    st.header("Same Date Uniqueness Details") 
    dup_details_pd_df = pd.DataFrame()
    with st.spinner("Loading duplicate record details..."):
        try:
            # First check if we have any UNIQUENESS rules for this brand in the summary
            all_summary_df = session.table("DQ_RESULTS").to_pandas()
            brand_summary_df = filter_df_by_brand(all_summary_df, data_brand)
            uniqueness_rules = brand_summary_df[brand_summary_df['RULE_TYPE'] == 'UNIQUENESS']
            
            st.subheader("UNIQUENESS Rules Summary")
            if not uniqueness_rules.empty:
                st.write(f"Found {len(uniqueness_rules)} UNIQUENESS rule(s) for {display_brand}:")
                st.dataframe(uniqueness_rules[['RULE_NAME', 'RULE_TYPE', 'INDICATOR', 'RESULT_VALUE', 'EXECUTION_TIMESTAMP']])
            else:
                st.info(f"No UNIQUENESS rules found for {display_brand} in the DQ_RESULTS table.")
            
            st.divider()
            st.subheader("Detailed Duplicate Records")
            
            # Now load the detailed duplicate records
            all_dup_details = session.table("DQ_DUPLICATE_DETAILS").to_pandas()
            dup_details_pd_df = filter_df_by_brand(all_dup_details, data_brand)
            
            if not dup_details_pd_df.empty:
                st.write(f"Found {len(dup_details_pd_df)} duplicate record detail(s):")
                st.dataframe(dup_details_pd_df)
            else:
                st.info(f"No detailed duplicate records found for {display_brand} in DQ_DUPLICATE_DETAILS table.")
                
        except Exception as e:
            st.error(f"Could not load duplicate record data: {e}")
            st.write("Available tables and columns for debugging:")
            try:
                # Show available tables for debugging
                tables_df = session.sql("SHOW TABLES LIKE 'DQ%'").to_pandas()
                st.write("Available DQ tables:")
                st.dataframe(tables_df)
            except Exception as table_err:
                st.error(f"Could not show tables: {table_err}")
            
    # *** Updated to use optimized AI summary for duplicate records ***
    generate_ai_summary_for_duplicates(
        data_df=dup_details_pd_df,
        brand_name=display_brand
    )

# --- TAB 3: Completeness Details ---
with tab3:
    st.header("Completeness Check Details")
    completeness_pd_df = pd.DataFrame()
    with st.spinner("Loading completeness details..."):
        try:
            all_completeness_df = session.table("DQ_DETAILS_COMPLETENESS").to_pandas()
            completeness_pd_df = filter_df_by_brand(all_completeness_df, data_brand)
            if not completeness_pd_df.empty:
                st.dataframe(completeness_pd_df)
            else:
                st.success(f"All completeness checks for {display_brand} were passed.")
        except Exception as e:
            st.error(f"Could not load data from DQ_DETAILS_COMPLETENESS: {e}")

    generate_ai_summary_for_tab(
        data_df=completeness_pd_df,
        analysis_type="Completeness Failures",
        brand_name=display_brand,
        relevant_cols=['EXPECTED_DATE', 'ACTUAL_MAX_DATE', 'DAYS_MISSING']
    )

# --- TAB 4: Sustained Trend Details ---
with tab4:
    st.header("Sustained Trend Details")
    details_pd_df = pd.DataFrame() 
    with st.spinner("Loading trend details..."):
        try:
            all_details_df = session.table("DQ_SUSTAINED_TREND_DETAILS").to_pandas()
            details_pd_df = filter_df_by_brand(all_details_df, data_brand)
            if not details_pd_df.empty:
                st.dataframe(details_pd_df)
            else:
                st.success(f"No sustained trends have been detected for {display_brand}.")
        except Exception as e:
            st.error(f"Could not load data from DQ_SUSTAINED_TREND_DETAILS table: {e}")
            
    generate_ai_summary_for_tab(
        data_df=details_pd_df,
        analysis_type="Sustained Trends",
        brand_name=display_brand,
        relevant_cols=['SEGMENT_VALUES', 'TREND_LENGTH']
    )

# --- TAB 5: Missing Data Details ---
with tab5:
    st.header("Missing Data & Nulls Details")
    missing_data_pd_df = pd.DataFrame()
    with st.spinner("Loading missing data and null violation details..."):
        try:
            all_missing_data_df = session.table("DQ_DETAILS_MISSING_DATA").to_pandas()
            missing_data_pd_df = filter_df_by_brand(all_missing_data_df, data_brand)
            
            if not missing_data_pd_df.empty:
                st.dataframe(missing_data_pd_df)
            else:
                st.success(f"No missing data or null violations have been detected for {display_brand}.")
        except Exception as e:
            st.error(f"Could not load data from DQ_DETAILS_MISSING_DATA: {e}")

    generate_ai_summary_for_tab(
        data_df=missing_data_pd_df,
        analysis_type="Missing Data and Null Violations",
        brand_name=display_brand,
        relevant_cols=['FAILURE_TYPE', 'FAILURE_DATE', 'DETAILS']
    )

# --- TAB 6: Spike / Dip Details ---
with tab6:
    st.header("Spike and Dip Details")
    st.subheader("Graphical View of Spikes & Dips")
    spike_dip_brand_df = pd.DataFrame()
    with st.spinner("Loading spike/dip details..."):
        try:
            all_spike_dip_details = session.table("DQ_DETAILS_SPIKE_DIP").to_pandas()
            spike_dip_brand_df = filter_df_by_brand(all_spike_dip_details, data_brand)
            if not spike_dip_brand_df.empty:
                def format_label_spike(segment_str):
                    try:
                        data = json.loads(segment_str)
                        return ", ".join([f"{k}: {v}" for k, v in data.items()])
                    except: return "Overall"
                spike_dip_brand_df["SEGMENT_LABEL"] = spike_dip_brand_df["SEGMENT_VALUES"].apply(format_label_spike)
                unique_segments = spike_dip_brand_df["SEGMENT_LABEL"].unique().tolist()
                segment_to_plot = st.selectbox("Select a Segment to Visualize:", options=unique_segments, key="spike_dip_chart_selector")
                if segment_to_plot:
                    selected_segment_json_str = spike_dip_brand_df[spike_dip_brand_df['SEGMENT_LABEL'] == segment_to_plot].iloc[0]['SEGMENT_VALUES']
                    source_table_name = "TRUE_NORTH_DAILY_TOTAL_CAJ_ORDERS_WEB_APP"
                    metric_col_name = "TOTAL_ORDERS"
                    with st.spinner(f"Fetching historical data for: {segment_to_plot}..."):
                        source_df = session.table(source_table_name)
                        if selected_segment_json_str != 'N/A':
                            segment_dict = json.loads(selected_segment_json_str)
                            for key, value in segment_dict.items():
                                source_df = source_df.filter(col(key) == value)
                        base_data_pd = source_df.select("ORDER_DATE", col(metric_col_name).alias("ACTUAL_DAILY_VALUE")).sort(col("ORDER_DATE").desc()).limit(180).to_pandas()
                        if not base_data_pd.empty:
                            segment_spikes = spike_dip_brand_df[spike_dip_brand_df['SEGMENT_VALUES'] == selected_segment_json_str]
                            base_data_pd['ORDER_DATE'] = pd.to_datetime(base_data_pd['ORDER_DATE'])
                            segment_spikes['EVENT_DATE'] = pd.to_datetime(segment_spikes['EVENT_DATE'])
                            merged_df = pd.merge(base_data_pd, segment_spikes[['EVENT_DATE']], left_on='ORDER_DATE', right_on='EVENT_DATE', how='left')
                            merged_df['SPIKE_OR_DIP'] = np.where(merged_df['EVENT_DATE'].notna(), merged_df['ACTUAL_DAILY_VALUE'], np.nan)
                            chart_df = merged_df.set_index('ORDER_DATE')[['ACTUAL_DAILY_VALUE', 'SPIKE_OR_DIP']]
                            st.line_chart(chart_df, color=["#1f77b4", "#d62728"])
                        else:
                            st.warning("No base historical data found to plot.")
                st.divider()
                st.subheader(f"Tabular Data for All '{display_brand}' Spikes/Dips")
                st.dataframe(spike_dip_brand_df)
            else:
                st.success(f"No spikes or dips have been detected for this brand.")
        except Exception as e:
            st.error(f"Could not load or plot spike/dip data: {e}", icon="🚨")

    generate_ai_summary_for_spike_dip(
        data_df=spike_dip_brand_df,
        brand_name=display_brand
    )

# --- TAB 7: Sigma & Anomaly Analysis (MERGED) ---
with tab7:
    st.header(f"Sigma & Anomaly Analysis for {display_brand}")
    analysis_choice = st.selectbox(
        "Choose Analysis Type:",
        ["3 Sigma Standard Deviation Bands", "2 to 3 Sigma Anomaly Detection (Snowflake ML)"],
        key="sigma_analysis_choice"
    )

    if analysis_choice == "3 Sigma Standard Deviation Bands":
        st.markdown("This chart displays the daily orders against the 60-day rolling average and standard deviation bands. Outliers beyond the 3-sigma bands are highlighted in red.")
        source_table_name = "TRUE_NORTH_DAILY_TOTAL_CAJ_ORDERS_WEB_APP"
        metric_col_name = "TOTAL_ORDERS"
        chart_pd_df = pd.DataFrame()
        sigma_outliers = pd.DataFrame()
        with st.spinner(f"Calculating 3-sigma bands for {display_brand}..."):
            try:
                source_df = session.table(source_table_name).filter(col("BRAND") == data_brand)
                window_spec = Window.orderBy("ORDER_DATE").rowsBetween(-59, 0)
                bands_df = source_df.with_column("AVG_60D", avg(col(metric_col_name)).over(window_spec)).with_column("STDDEV_60D", stddev(col(metric_col_name)).over(window_spec))
                final_bands_df = bands_df.with_column("PLUS_1_SIGMA", col("AVG_60D") + col("STDDEV_60D")).with_column("MINUS_1_SIGMA", col("AVG_60D") - col("STDDEV_60D")).with_column("PLUS_2_SIGMA", col("AVG_60D") + (2 * col("STDDEV_60D"))).with_column("MINUS_2_SIGMA", col("AVG_60D") - (2 * col("STDDEV_60D"))).with_column("PLUS_3_SIGMA", col("AVG_60D") + (3 * col("STDDEV_60D"))).with_column("MINUS_3_SIGMA", col("AVG_60D") - (3 * col("STDDEV_60D")))
                
                chart_data_sp_df = final_bands_df.select(
                    "ORDER_DATE",
                    col(metric_col_name).alias("Actual Orders"),
                    "AVG_60D",
                    "PLUS_1_SIGMA", "MINUS_1_SIGMA",
                    "PLUS_2_SIGMA", "MINUS_2_SIGMA",
                    "PLUS_3_SIGMA", "MINUS_3_SIGMA"
                ).sort(col("ORDER_DATE").desc()).limit(365)
                
                chart_pd_df = chart_data_sp_df.to_pandas()
                
                if not chart_pd_df.empty:
                    chart_pd_df = chart_pd_df.rename(columns={
                        "AVG_60D": "60-Day Rolling Average",
                        "PLUS_1_SIGMA": "+1 Sigma", "MINUS_1_SIGMA": "-1 Sigma",
                        "PLUS_2_SIGMA": "+2 Sigma", "MINUS_2_SIGMA": "-2 Sigma",
                        "PLUS_3_SIGMA": "+3 Sigma", "MINUS_3_SIGMA": "-3 Sigma"
                    })
                    
                    sigma_outliers = chart_pd_df[
                        (chart_pd_df['Actual Orders'] > chart_pd_df['+3 Sigma']) |
                        (chart_pd_df['Actual Orders'] < chart_pd_df['-3 Sigma'])
                    ].copy()
                    sigma_outliers['Zone'] = np.where(sigma_outliers['Actual Orders'] > sigma_outliers['+3 Sigma'], 'Upper Outlier (>3σ)', 'Lower Outlier (<3σ)')

                    bands_to_plot = ["+1 Sigma", "-1 Sigma", "+2 Sigma", "-2 Sigma", "+3 Sigma", "-3 Sigma"]
                    melted_df = chart_pd_df.melt(id_vars=['ORDER_DATE'], value_vars=bands_to_plot, var_name='Band', value_name='Value')
                    
                    actual_line = alt.Chart(chart_pd_df).mark_line(color='blue').encode(x=alt.X('ORDER_DATE:T', title='Date'), y=alt.Y('Actual Orders:Q', title='Total Orders'), tooltip=['ORDER_DATE:T', 'Actual Orders:Q'])
                    average_line = alt.Chart(chart_pd_df).mark_line(color='black', strokeWidth=1.5, opacity=0.9).encode(x=alt.X('ORDER_DATE:T'), y=alt.Y('60-Day Rolling Average:Q'), tooltip=['ORDER_DATE:T', alt.Tooltip('60-Day Rolling Average:Q', format='.2f')])
                    sigma_lines = alt.Chart(melted_df).mark_line(strokeDash=[5,5], opacity=0.8).encode(x=alt.X('ORDER_DATE:T', title='Date'), y=alt.Y('Value:Q', title='Total Orders'), color=alt.Color('Band:N', scale=alt.Scale(domain=bands_to_plot, range=['#2ca02c', '#2ca02c', '#ff7f0e', '#ff7f0e', '#d62728', '#d62728'])), tooltip=['ORDER_DATE:T', 'Value:Q', 'Band:N'])
                    highlight_points = alt.Chart(sigma_outliers).mark_point(size=80, filled=True, color='red', opacity=1, shape='diamond').encode(x=alt.X('ORDER_DATE:T'), y=alt.Y('Actual Orders:Q'), tooltip=['ORDER_DATE:T', 'Actual Orders:Q', 'Zone:N'])
                    final_chart = (actual_line + average_line + sigma_lines + highlight_points).interactive()
                    st.altair_chart(final_chart, use_container_width=True)
                    
                    if not sigma_outliers.empty:
                        st.subheader("Logged 3-Sigma Outliers (Red Indicator)")
                        st.dataframe(sigma_outliers[['ORDER_DATE', 'Actual Orders', '60-Day Rolling Average', '+3 Sigma', '-3 Sigma', 'Zone']])
                    else:
                        st.success("No 3-sigma outliers detected for the selected period.")
                else:
                    st.warning(f"No historical data found for {display_brand} to calculate bands.")
            except Exception as e:
                st.error(f"An error occurred while generating the 3-sigma chart: {e}")

        generate_ai_summary_for_tab(data_df=sigma_outliers, analysis_type="3 Sigma Outliers", brand_name=display_brand, relevant_cols=['ORDER_DATE', 'Actual Orders', '+3 Sigma', '-3 Sigma', 'Zone'])
        
    elif analysis_choice == "2 to 3 Sigma Anomaly Detection (Snowflake ML)":
        st.markdown("This chart identifies 'warning' data points that fall between the 2nd and 3rd standard deviations from the 60-day rolling average. These points are highlighted in amber.")
        source_table_name = "TRUE_NORTH_DAILY_TOTAL_CAJ_ORDERS_WEB_APP"
        metric_col_name = "TOTAL_ORDERS"
        anomalies_pd_df = pd.DataFrame()

        with st.spinner(f"Calculating 2-to-3 sigma warnings for {display_brand}..."):
            try:
                source_df = session.table(source_table_name).filter(col("BRAND") == data_brand)
                window_spec = Window.orderBy("ORDER_DATE").rowsBetween(-59, 0)
                bands_df = source_df.with_column("AVG_60D", avg(col(metric_col_name)).over(window_spec)).with_column("STDDEV_60D", stddev(col(metric_col_name)).over(window_spec))
                final_bands_df = bands_df.with_column("PLUS_2_SIGMA", col("AVG_60D") + (2 * col("STDDEV_60D"))).with_column("MINUS_2_SIGMA", col("AVG_60D") - (2 * col("STDDEV_60D"))).with_column("PLUS_3_SIGMA", col("AVG_60D") + (3 * col("STDDEV_60D"))).with_column("MINUS_3_SIGMA", col("AVG_60D") - (3 * col("STDDEV_60D")))
                chart_data_sp_df = final_bands_df.sort(col("ORDER_DATE").desc()).limit(365)
                chart_pd_df = chart_data_sp_df.to_pandas()

                if not chart_pd_df.empty:
                    chart_pd_df = chart_pd_df.rename(columns={metric_col_name: "Actual Orders", "AVG_60D": "60-Day Rolling Average", "PLUS_2_SIGMA": "+2 Sigma", "MINUS_2_SIGMA": "-2 Sigma", "PLUS_3_SIGMA": "+3 Sigma", "MINUS_3_SIGMA": "-3 Sigma"})
                    upper_zone = (chart_pd_df['Actual Orders'] > chart_pd_df['+2 Sigma']) & (chart_pd_df['Actual Orders'] <= chart_pd_df['+3 Sigma'])
                    lower_zone = (chart_pd_df['Actual Orders'] < chart_pd_df['-2 Sigma']) & (chart_pd_df['Actual Orders'] >= chart_pd_df['-3 Sigma'])
                    anomalies_pd_df = chart_pd_df[upper_zone | lower_zone].copy()
                    anomalies_pd_df['Zone'] = np.where(anomalies_pd_df['Actual Orders'] > anomalies_pd_df['60-Day Rolling Average'], 'Upper Warning Zone', 'Lower Warning Zone')
                    bands_to_plot = ["+2 Sigma", "-2 Sigma", "+3 Sigma", "-3 Sigma"]
                    melted_df = chart_pd_df.melt(id_vars=['ORDER_DATE'], value_vars=bands_to_plot, var_name='Band', value_name='Value')
                    actual_line = alt.Chart(chart_pd_df).mark_line(color='blue').encode(x=alt.X('ORDER_DATE:T', title='Date'), y=alt.Y('Actual Orders:Q', title='Total Orders'), tooltip=['ORDER_DATE:T', 'Actual Orders:Q'])
                    average_line = alt.Chart(chart_pd_df).mark_line(color='black', strokeWidth=1.5, opacity=0.9).encode(x=alt.X('ORDER_DATE:T'), y=alt.Y('60-Day Rolling Average:Q'), tooltip=['ORDER_DATE:T', alt.Tooltip('60-Day Rolling Average:Q', format='.2f')])
                    sigma_lines = alt.Chart(melted_df).mark_line(strokeDash=[5,5], opacity=0.8).encode(x=alt.X('ORDER_DATE:T'), y=alt.Y('Value:Q'), color=alt.Color('Band:N', scale=alt.Scale(domain=["+2 Sigma", "-2 Sigma", "+3 Sigma", "-3 Sigma"], range=['#ff7f0e', '#ff7f0e', '#d62728', '#d62728'])), tooltip=['ORDER_DATE:T', 'Value:Q', 'Band:N'])
                    highlight_points = alt.Chart(anomalies_pd_df).mark_point(size=80, filled=True, color='orange', opacity=1, shape='diamond').encode(x=alt.X('ORDER_DATE:T'), y=alt.Y('Actual Orders:Q'), tooltip=['ORDER_DATE:T', 'Actual Orders:Q', 'Zone:N'])
                    final_chart = (actual_line + average_line + sigma_lines + highlight_points).interactive()
                    st.altair_chart(final_chart, use_container_width=True)

                    if not anomalies_pd_df.empty:
                        st.subheader("Logged Warning Zone Events")
                        st.dataframe(anomalies_pd_df[['ORDER_DATE', 'Actual Orders', '60-Day Rolling Average', 'STDDEV_60D', 'Zone']])
                    else:
                        st.success("No data points found in the 2-to-3 sigma warning zone for the selected period.")
                else:
                    st.warning(f"No historical data found for {display_brand} to calculate bands.")
            except Exception as e:
                st.error(f"An error occurred while generating the 2-to-3 sigma warning chart: {e}")

        generate_ai_summary_for_tab(data_df=anomalies_pd_df, analysis_type="2-to-3 Sigma Warning Zone Events", brand_name=display_brand, relevant_cols=['ORDER_DATE', 'Actual Orders', '60-Day Rolling Average', 'Zone'])

# --- Global AI Help Section at the Bottom ---
st.divider()
st.header("AI Rule Explanations")
help_question = st.text_input("Ask what a rule means (e.g., 'What is a sustained trend?')", key="main_help_question")
if st.button("Explain Rule", key="main_explain_button"):
    if help_question:
        with st.spinner("AI is thinking..."):
            st.info(call_cortex_llm(get_help_prompt(help_question)))
    else:
        st.warning("Please enter a question about a rule.")