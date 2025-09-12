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
    # Handle empty dataframes
    if df.empty:
        return df
    
    # For COMPLETENESS details table, we need different logic since it doesn't have JSON columns
    if 'RULE_NAME' in df.columns and 'DAYS_MISSING' in df.columns and 'EXPECTED_DATE' in df.columns:
        # This is likely the DQ_DETAILS_COMPLETENESS table
        if brand_data_value == "__GLOBAL__":
            # For global view, show all completeness records
            return df.copy()
        else:
            # For brand-specific view, filter by rule name containing the brand
            name_filter = df['RULE_NAME'].str.contains(brand_data_value, case=False, na=False)
            return df[name_filter].copy()
    
    json_col_candidates = ['DETAILS', 'DUPLICATE_VALUES', 'SEGMENT_VALUES', 'SEGMENT_VALUE']
    json_col_name = next((col for col in json_col_candidates if col in df.columns), None)

    # --- NEW LOGIC FOR UNCATEGORIZED FAILURES ---
    if brand_data_value == "__GLOBAL__":
        if not json_col_name:
            return df.copy() # Return all data if no JSON column exists

        def is_json_uncategorized(json_str):
            try:
                if isinstance(json_str, str):
                    if not json_str.strip() or json_str == '{}': 
                        return True
                    data = json.loads(json_str)
                elif isinstance(json_str, dict):
                    if not json_str: 
                        return True
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
    
    # Special handling for Historical Completeness rules - they should show for WSJ brand
    historical_completeness_filter = pd.Series([False] * len(df), index=df.index)
    if 'RULE_TYPE' in df.columns and brand_data_value == "WSJ":
        # Show Historical Completeness rules for WSJ since they monitor WSJ data
        historical_completeness_filter = (df['RULE_TYPE'] == 'HISTORICAL_COMPLETENESS')
    
    global_summary_filter = pd.Series([False] * len(df), index=df.index)
    if 'RULE_TYPE' in df.columns:
        # Show UNIQUENESS and HISTORICAL_COMPLETENESS rules - try multiple approaches to find them
        if brand_data_value == "__GLOBAL__":
            # For global view, show all UNIQUENESS and HISTORICAL_COMPLETENESS rules
            global_summary_filter = (df['RULE_TYPE'].isin(['UNIQUENESS', 'HISTORICAL_COMPLETENESS']))
        else:
            # For brand-specific view, show rules that might be related to this brand
            relevant_rule_types = (df['RULE_TYPE'].isin(['UNIQUENESS', 'HISTORICAL_COMPLETENESS']))
            # Try different ways to match brand: rule name contains brand, or show all relevant rules
            brand_in_name = df['RULE_NAME'].str.contains(brand_data_value, case=False, na=False)
            # For now, show ALL relevant rules on brand pages to debug
            global_summary_filter = relevant_rule_types

    json_filter = pd.Series([False] * len(df), index=df.index)
    if json_col_name:
        def check_brand_in_json(json_str):
            try:
                if isinstance(json_str, str):
                    if not json_str.strip(): 
                        return False
                    data = json.loads(json_str)
                elif isinstance(json_str, dict):
                    data = json_str
                else:
                    return False
                return data.get("BRAND") == brand_data_value
            except (json.JSONDecodeError, TypeError, AttributeError):
                return False
                return False
        json_filter = df[json_col_name].apply(check_brand_in_json)

    final_filter = name_filter | json_filter | historical_completeness_filter
    return df[final_filter].copy()

# --- AI Helper Functions (MODIFIED FOR CONTEXTUAL ANALYSIS) ---
RULE_DEFINITIONS = """
- **Uniqueness / Duplicate Check**: Verifies that records are unique. This can be a simple SQL-based check for duplicates on a given day or a more advanced check that finds records where multiple columns are identical, with special handling for case-insensitivity ('WSJ' = 'wsj') and numeric rounding.
- **Historical Completeness**: Monitors record count consistency over 60-day rolling windows by calculating expected changes (removing the 61st day count and adding the newest day count) and comparing against actual cumulative counts. Flags discrepancies outside tolerance thresholds.
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

def generate_ai_insights_for_tab(data_df: pd.DataFrame, analysis_type: str, brand_name: str, relevant_cols: list, full_data_df: pd.DataFrame = None):
    """Creates enhanced AI insights for any data scenario - success, failure, or mixed results."""
    st.divider()
    st.subheader("AI-Powered Analysis")
    
    # Create contextual button text
    button_text = f"Get AI Insights on {analysis_type}"
    button_key = f"ai_insights_{analysis_type.replace(' ', '_').replace('/', '_')}"
    
    if st.button(button_text, key=button_key):
        with st.spinner(f"AI is analyzing {analysis_type.lower()} data..."):
            # Determine the data scenario
            has_data = data_df is not None and not data_df.empty
            has_full_data = full_data_df is not None and not full_data_df.empty
            
            if has_data or has_full_data:
                # Use appropriate dataset for analysis
                analysis_data = data_df if has_data else full_data_df
                
                # Determine data quality status
                if 'STATUS' in analysis_data.columns:
                    pass_count = len(analysis_data[analysis_data['STATUS'] == 'PASS'])
                    fail_count = len(analysis_data[analysis_data['STATUS'] == 'FAIL'])
                    error_count = len(analysis_data[analysis_data['STATUS'] == 'ERROR'])
                    total_count = len(analysis_data)
                elif 'INDICATOR' in analysis_data.columns:
                    pass_count = len(analysis_data[analysis_data['INDICATOR'] == 'PASS'])
                    fail_count = len(analysis_data[analysis_data['INDICATOR'] == 'FAIL'])
                    error_count = len(analysis_data[analysis_data['INDICATOR'] == 'ERROR'])
                    total_count = len(analysis_data)
                else:
                    # For data without explicit status, assume issues if data exists
                    fail_count = len(analysis_data)
                    pass_count = 0
                    error_count = 0
                    total_count = len(analysis_data)
                
                # Create enhanced prompt based on scenario
                if fail_count == 0 and error_count == 0:
                    # Success scenario
                    scenario = "SUCCESS"
                    scenario_description = f"All {total_count} {analysis_type.lower()} checks are passing"
                elif fail_count > 0 and pass_count > 0:
                    # Mixed scenario
                    scenario = "MIXED"
                    scenario_description = f"Mixed results: {pass_count} passing, {fail_count} failing"
                else:
                    # Failure scenario
                    scenario = "FAILURE"
                    scenario_description = f"Issues detected: {fail_count} failures, {error_count} errors"
                
                # Prepare data for analysis
                analysis_cols = [col for col in relevant_cols if col in analysis_data.columns]
                data_for_prompt = analysis_data[analysis_cols].to_json(orient='records', date_format='iso')
                
                # Create enhanced prompt
                enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'. 
                
ANALYSIS CONTEXT:
- Rule Type: {analysis_type}
- Data Scenario: {scenario_description}
- Total Records Analyzed: {total_count}

INSTRUCTIONS:
Based on the {scenario.lower()} scenario, provide insights that include:

1. **Overall Assessment**: What does this data tell us about {analysis_type.lower()} for {brand_name}?

2. **Key Findings**: Highlight the most important patterns, trends, or issues.

3. **Business Impact**: What are the implications for data quality and business operations?

4. **Recommendations**: 
   - If everything is good: How to maintain this quality level
   - If there are issues: Immediate actions and investigation steps
   - If mixed results: Prioritization and focus areas

5. **Monitoring Strategy**: What should be watched going forward?

DATA TO ANALYZE:
{data_for_prompt}

Be concise but comprehensive. Focus on actionable insights."""

                summary = call_cortex_llm(enhanced_prompt)
                st.markdown(summary)
                
                # Add summary metrics
                if total_count > 0:
                    st.info(f"📊 **Analysis Summary**: {scenario_description} out of {total_count} total records")
                
            else:
                # No data scenario - still provide value
                no_data_prompt = f"""You are a data quality analyst for '{brand_name}'. 
                
There is currently no {analysis_type.lower()} data to analyze, which could mean:
1. All systems are operating perfectly (ideal scenario)
2. No rules have been executed yet
3. Data collection issues

Provide insights on:
1. What this lack of data typically indicates for {analysis_type.lower()}
2. Best practices for monitoring {analysis_type.lower()}
3. Recommended proactive measures for {brand_name}
4. What to watch for when data becomes available

Be encouraging but informative about maintaining data quality standards."""
                
                summary = call_cortex_llm(no_data_prompt)
                st.success(f"✅ No {analysis_type.lower()} issues detected for {brand_name}")
                st.markdown(summary)

def generate_ai_insights_for_duplicates(data_df: pd.DataFrame, brand_name: str):
    """Creates enhanced AI insights specifically for duplicate records with universal functionality."""
    st.divider()
    st.subheader("AI-Powered Analysis")
    if st.button("Get AI Insights on Duplicate Records", key="ai_insights_duplicate_records"):
        with st.spinner("AI is analyzing duplicate records data..."):
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
                        
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'. 
                        
DUPLICATE RECORDS ANALYSIS:
- Total duplicate entries found: {total_records}
- Total affected records: {int(total_duplicate_count)}
- Average duplicates per entry: {round(avg_duplicates, 2)}
- Maximum duplicates for single entry: {int(max_duplicates)}
- Minimum duplicates for single entry: {int(min_duplicates)}
- Unique dates affected: {unique_dates if unique_dates else "Not available"}

ANALYSIS SCOPE: Large dataset ({total_records} entries) - providing statistical summary.

Provide comprehensive insights on:
1. **Scale Assessment**: How significant is this duplication issue for {brand_name}?
2. **Data Quality Impact**: What do these patterns indicate about data collection processes?
3. **Business Risk**: Potential operational and financial implications
4. **Root Cause Analysis**: Likely sources of these duplications
5. **Immediate Actions**: Prioritized steps for investigation and resolution
6. **Prevention Strategy**: Long-term measures to prevent future duplications
7. **Monitoring Recommendations**: Ongoing quality checks to implement

Focus on actionable business insights and practical next steps."""
                        
                    except Exception as e:
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'. 
                        
There are {total_records} duplicate record entries requiring immediate attention.
                        
Provide insights on:
1. The business significance of having {total_records} duplicate entries
2. Potential operational and financial impact 
3. Immediate investigation and remediation actions
4. Prevention strategies for future occurrences
5. Monitoring recommendations
                        
Focus on actionable guidance for this scale of duplication issue."""
                else:
                    # For smaller datasets, analyze individual records
                    relevant_cols = ['DUPLICATE_VALUES', 'DUPLICATE_COUNT'] if 'DUPLICATE_COUNT' in data_df.columns else list(data_df.columns)
                    relevant_cols = [col for col in relevant_cols if col in data_df.columns]
                    data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                    
                    enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                    
DUPLICATE RECORDS ANALYSIS:
- Dataset Size: {total_records} duplicate entries (detailed analysis)

Analyze the following duplicate records data and provide insights on:
1. **Pattern Analysis**: What patterns do you see in the duplications?
2. **Severity Assessment**: How critical are these duplications?
3. **Business Impact**: Effects on data reliability and operations
4. **Investigation Steps**: Specific actions to understand root causes
5. **Resolution Strategy**: Step-by-step remediation approach
6. **Prevention Measures**: How to avoid future duplications

Data to analyze:
{data_for_prompt}

Provide practical, actionable insights focused on data quality improvement."""
                
                summary = call_cortex_llm(enhanced_prompt)
                st.markdown(summary)
                st.info(f"📊 **Analysis Summary**: Found {total_records} duplicate record entries affecting data quality")
            else:
                # No duplicates found - success scenario
                success_prompt = f"""You are a data quality analyst for '{brand_name}'.
                
DUPLICATE RECORDS STATUS: No duplicate records detected - excellent data quality!

Provide insights on:
1. **Quality Achievement**: What this clean state indicates about {brand_name}'s data processes
2. **Maintenance Strategy**: How to sustain this high data quality standard
3. **Proactive Monitoring**: Best practices for ongoing duplicate prevention
4. **System Health**: What this suggests about data collection and validation processes
5. **Future Vigilance**: Key indicators to monitor for potential duplication issues

Focus on maintaining and building upon this excellent data quality foundation."""
                
                summary = call_cortex_llm(success_prompt)
                st.success(f"✅ No duplicate records detected for {brand_name} - excellent data quality!")
                st.markdown(summary)

def generate_ai_insights_for_sustained_trends(data_df: pd.DataFrame, brand_name: str):
    """Creates enhanced AI insights specifically for sustained trend events with universal functionality."""
    st.divider()
    st.subheader("AI-Powered Analysis")
    if st.button("Get AI Insights on Sustained Trends", key="ai_insights_sustained_trends"):
        with st.spinner("AI is analyzing sustained trend data..."):
            if data_df is not None and not data_df.empty:
                total_trends = len(data_df)
                
                if total_trends > 100:
                    # For large datasets, provide summary statistics instead of individual trends
                    try:
                        # Calculate summary statistics
                        if 'TREND_LENGTH' in data_df.columns:
                            avg_trend_length = data_df['TREND_LENGTH'].mean()
                            max_trend_length = data_df['TREND_LENGTH'].max()
                            min_trend_length = data_df['TREND_LENGTH'].min()
                            total_trend_days = data_df['TREND_LENGTH'].sum()
                        else:
                            avg_trend_length = max_trend_length = min_trend_length = total_trend_days = "Not available"
                        
                        # Get unique dates and segments
                        unique_dates = "Not available"
                        date_columns = [col for col in data_df.columns if 'DATE' in col.upper()]
                        if date_columns:
                            unique_dates = data_df[date_columns[0]].nunique()
                        
                        unique_segments = "Not available"
                        if 'SEGMENT_VALUES' in data_df.columns:
                            unique_segments = data_df['SEGMENT_VALUES'].nunique()
                        
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                        
SUSTAINED TREND ANALYSIS:
- Total sustained trends detected: {total_trends}
- Average trend length: {round(avg_trend_length, 1) if isinstance(avg_trend_length, (int, float)) else avg_trend_length} days
- Longest trend: {int(max_trend_length) if isinstance(max_trend_length, (int, float)) else max_trend_length} days
- Shortest trend: {int(min_trend_length) if isinstance(min_trend_length, (int, float)) else min_trend_length} days
- Total trend days: {int(total_trend_days) if isinstance(total_trend_days, (int, float)) else total_trend_days} days
- Unique dates affected: {unique_dates}
- Unique segments affected: {unique_segments}

ANALYSIS SCOPE: Large dataset ({total_trends} trends) - providing statistical summary.

Provide comprehensive insights on:
1. **Trend Significance**: What does detecting {total_trends} sustained trends indicate for {brand_name}?
2. **Pattern Analysis**: What do these trend lengths and frequencies suggest about business dynamics?
3. **Business Impact**: Operational implications of sustained growth or decline patterns
4. **Market Indicators**: What sustained trends typically signal in order volume data
5. **Risk Assessment**: Potential concerns with prolonged directional movements
6. **Strategic Insights**: How to leverage positive trends and address negative ones
7. **Monitoring Strategy**: Recommended actions for ongoing trend surveillance

Focus on actionable business insights for trend management and strategic planning."""
                        
                    except Exception as e:
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                        
There are {total_trends} sustained trend events indicating significant directional patterns in order data.
                        
Provide insights on:
1. The business significance of {total_trends} sustained trend events
2. Potential operational impact of sustained directional movements
3. Immediate analysis and strategic response actions
4. Long-term trend monitoring recommendations
                        
Focus on actionable guidance for trend-based decision making."""
                else:
                    # For smaller datasets, analyze individual trends
                    relevant_cols = ['SEGMENT_VALUES', 'TREND_LENGTH', 'TREND_START_DATE', 'TREND_END_DATE']
                    relevant_cols = [col for col in relevant_cols if col in data_df.columns]
                    data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                    
                    enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                    
SUSTAINED TREND ANALYSIS:
- Dataset Size: {total_trends} sustained trends (detailed analysis)

Analyze the following sustained trend data and provide insights on:
1. **Trend Patterns**: What specific patterns emerge from these sustained movements?
2. **Duration Analysis**: How concerning or promising are these trend lengths?
3. **Business Impact**: Effects on order volume stability and growth
4. **Segment Analysis**: Which segments show the most significant trends?
5. **Strategic Response**: Specific actions for positive vs negative trends
6. **Monitoring Recommendations**: How to track and respond to future trends

Data to analyze:
{data_for_prompt}

Provide practical insights for trend-based business strategy and operations."""
                
                summary = call_cortex_llm(enhanced_prompt)
                st.markdown(summary)
                st.info(f"📊 **Analysis Summary**: Detected {total_trends} sustained trend events requiring strategic attention")
            else:
                # No sustained trends - success scenario
                success_prompt = f"""You are a data quality analyst for '{brand_name}'.
                
SUSTAINED TREND STATUS: No sustained trends detected - balanced order patterns!

Provide insights on:
1. **Pattern Stability**: What this balanced state indicates about {brand_name}'s market performance
2. **Market Position**: Benefits of stable, non-trending order patterns
3. **Operational Advantages**: How balanced patterns support business operations
4. **Growth Opportunities**: Strategies to generate positive sustained trends
5. **Monitoring Strategy**: Best practices to detect early trend formations
6. **Competitive Analysis**: What stable patterns suggest about market positioning

Focus on leveraging stability while preparing for strategic growth initiatives."""
                
                summary = call_cortex_llm(success_prompt)
                st.success(f"✅ No sustained trends detected for {brand_name} - balanced order patterns!")
                st.markdown(summary)

def generate_ai_insights_for_spike_dip(data_df: pd.DataFrame, brand_name: str):
    """Creates enhanced AI insights specifically for spike and dip events with universal functionality."""
    st.divider()
    st.subheader("AI-Powered Analysis")
    if st.button("Get AI Insights on Spike and Dip Events", key="ai_insights_spike_dip_events"):
        with st.spinner("AI is analyzing spike and dip events data..."):
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
                        
                        # Get unique dates and segments
                        unique_dates = "Not available"
                        date_columns = [col for col in data_df.columns if 'DATE' in col.upper()]
                        if date_columns:
                            unique_dates = data_df[date_columns[0]].nunique()
                        
                        unique_segments = "Not available"
                        if 'SEGMENT_VALUES' in data_df.columns:
                            unique_segments = data_df['SEGMENT_VALUES'].nunique()
                        
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                        
SPIKE AND DIP EVENTS ANALYSIS:
- Total volatility events: {total_events}
- Spikes (positive changes): {spike_count}
- Dips (negative changes): {dip_count}
- Average percent change: {round(avg_change, 2) if isinstance(avg_change, (int, float)) else avg_change}%
- Maximum spike: {round(max_spike, 2) if isinstance(max_spike, (int, float)) else max_spike}%
- Maximum dip: {round(max_dip, 2) if isinstance(max_dip, (int, float)) else max_dip}%
- Volatility level (std dev): {round(std_change, 2) if isinstance(std_change, (int, float)) else std_change}%
- Unique dates affected: {unique_dates}
- Unique segments affected: {unique_segments}

ANALYSIS SCOPE: Large dataset ({total_events} events) - providing statistical summary.

Provide comprehensive insights on:
1. **Volatility Assessment**: What does this level of order fluctuation indicate for {brand_name}?
2. **Pattern Analysis**: Balance between spikes vs dips and what this suggests
3. **Business Impact**: Operational implications of this volatility level
4. **Market Factors**: Potential external drivers of these patterns
5. **Risk Evaluation**: Critical thresholds and concerning trends
6. **Operational Response**: Recommended monitoring and response strategies
7. **Stability Measures**: Actions to reduce harmful volatility

Focus on actionable business insights for order volume management."""
                        
                    except Exception as e:
                        enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                        
There are {total_events} spike and dip events indicating significant order volatility.
                        
Provide insights on:
1. The business significance of {total_events} volatility events
2. Potential operational impact of this order fluctuation level
3. Immediate analysis and monitoring actions needed
4. Stability improvement recommendations
                        
Focus on actionable guidance for managing order volatility."""
                else:
                    # For smaller datasets, analyze individual events
                    relevant_cols = ['EVENT_DATE', 'METRIC_VALUE', 'PREVIOUS_METRIC_VALUE', 'PERCENT_CHANGE']
                    relevant_cols = [col for col in relevant_cols if col in data_df.columns]
                    data_for_prompt = data_df[relevant_cols].to_json(orient='records', date_format='iso')
                    
                    enhanced_prompt = f"""You are a data quality analyst for '{brand_name}'.
                    
SPIKE AND DIP EVENTS ANALYSIS:
- Dataset Size: {total_events} volatility events (detailed analysis)

Analyze the following spike and dip data and provide insights on:
1. **Event Patterns**: What specific patterns emerge from these volatility events?
2. **Severity Assessment**: How concerning are these fluctuations?
3. **Business Impact**: Effects on order processing and operations
4. **Trend Analysis**: Direction and frequency of volatility
5. **Response Strategy**: Specific actions for each type of event
6. **Prevention Measures**: Ways to stabilize order volumes

Data to analyze:
{data_for_prompt}

Provide practical insights for order volume stability and business continuity."""
                
                summary = call_cortex_llm(enhanced_prompt)
                st.markdown(summary)
                st.info(f"📊 **Analysis Summary**: Detected {total_events} order volatility events requiring monitoring")
            else:
                # No volatility events - success scenario
                success_prompt = f"""You are a data quality analyst for '{brand_name}'.
                
SPIKE AND DIP STATUS: No significant order volatility detected - excellent stability!

Provide insights on:
1. **Stability Achievement**: What this consistent order pattern indicates about {brand_name}'s market position
2. **Operational Excellence**: How stable orders benefit business operations
3. **Monitoring Strategy**: Best practices to maintain this order stability
4. **Early Warning Systems**: Key indicators to watch for potential volatility
5. **Growth Opportunities**: How stability enables strategic planning
6. **Competitive Advantage**: What consistent performance means in the market

Focus on leveraging and maintaining this excellent order stability."""
                
                summary = call_cortex_llm(success_prompt)
                st.success(f"✅ No significant order volatility detected for {brand_name} - excellent stability!")
                st.markdown(summary)

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Summary", 
    "Uniqueness / Duplicate Details", 
    "Completeness Details",
    "Historical Completeness Details",
    "Sustained Trend Details", 
    "Missing Data Details", 
    "Spike / Dip Details", 
    "Sigma & Anomaly Analysis",
    "Negative Value Details"
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
                # Get RULE_TYPE values from the actual data (not RULE_NAME)
                all_rule_types = all_summary_df['RULE_TYPE'].unique().tolist()
                # Filter out system-level rule types that aren't actual data quality rules
                business_rule_types = sorted([rt for rt in all_rule_types if rt not in ['SYSTEM', 'SYSTEM_ERROR']])
                
                all_option = "All Rule Types"
                rule_type_options = [all_option] + business_rule_types
                user_selected_types = st.multiselect("Filter by Rule Type:", options=rule_type_options, default=[all_option], key="summary_type_filter")
                selected_rule_types = business_rule_types if all_option in user_selected_types else user_selected_types

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
                
                # Add UNIQUENESS and HISTORICAL_COMPLETENESS rule metrics
                uniqueness_count = len(filtered_summary_df[filtered_summary_df['RULE_TYPE'] == 'UNIQUENESS'])
                uniqueness_failures = len(filtered_summary_df[
                    (filtered_summary_df['RULE_TYPE'] == 'UNIQUENESS') & 
                    (filtered_summary_df['INDICATOR'] == 'FAIL')
                ])
                
                hist_completeness_count = len(filtered_summary_df[filtered_summary_df['RULE_TYPE'] == 'HISTORICAL_COMPLETENESS'])
                hist_completeness_failures = len(filtered_summary_df[
                    (filtered_summary_df['RULE_TYPE'] == 'HISTORICAL_COMPLETENESS') & 
                    (filtered_summary_df['INDICATOR'] == 'FAIL')
                ])
                
                metric_cols = st.columns(5)
                metric_cols[0].metric("Total Rules Executed", f"{total_runs}")
                metric_cols[1].metric("Pass Rate", f"{pass_rate:.2f}%")
                metric_cols[2].metric("Total Failures/Errors", f"{fail_count + error_count}", delta=f"{fail_count + error_count} issues", delta_color="inverse")
                metric_cols[3].metric("UNIQUENESS Rules", f"{uniqueness_count}", help=f"Failed: {uniqueness_failures}")
                metric_cols[4].metric("HISTORICAL_COMPLETENESS Rules", f"{hist_completeness_count}", help=f"Failed: {hist_completeness_failures}")
                
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
                
            generate_ai_insights_for_tab(
                data_df=filtered_summary_df[filtered_summary_df['INDICATOR'] == 'FAIL'],
                analysis_type="Overall Quality Results",
                brand_name=display_brand,
                relevant_cols=['RULE_NAME', 'RULE_TYPE', 'RESULT_VALUE'],
                full_data_df=filtered_summary_df
            )
        except Exception as e:
            st.error(f"Could not load summary data: {e}")

# --- TAB 2: Uniqueness / Duplicate Details ---
with tab2:
    st.header("Uniqueness / Duplicate Details") 
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
            
    # *** Updated to use enhanced AI insights for duplicate records ***
    generate_ai_insights_for_duplicates(
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
                st.subheader("Completeness Check Results")
                
                # Display the data with better formatting
                st.dataframe(
                    completeness_pd_df,
                    use_container_width=True,
                    column_config={
                        "EXECUTION_TIMESTAMP": st.column_config.DatetimeColumn(
                            "Execution Time",
                            format="YYYY-MM-DD HH:mm:ss"
                        ),
                        "EXPECTED_DATE": st.column_config.TextColumn(
                            "Expected Date"
                        ),
                        "ACTUAL_MAX_DATE": st.column_config.TextColumn(
                            "Actual Max Date"
                        ),
                        "DAYS_MISSING": st.column_config.NumberColumn(
                            "Count/Days Missing",
                            help="Number of records found or days missing"
                        )
                    }
                )
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Checks", len(completeness_pd_df))
                with col2:
                    # Count checks with actual issues (non-zero days missing or count > 0)
                    issues_count = len(completeness_pd_df[
                        (completeness_pd_df['DAYS_MISSING'] > 0) & 
                        (completeness_pd_df['ACTUAL_MAX_DATE'] != 'N/A - Count Check')
                    ])
                    st.metric("Data Gap Issues", issues_count)
                with col3:
                    # Count completeness violations (where DAYS_MISSING represents count > 0)
                    violations_count = len(completeness_pd_df[
                        (completeness_pd_df['ACTUAL_MAX_DATE'] == 'N/A - Count Check') & 
                        (completeness_pd_df['DAYS_MISSING'] > 0)
                    ])
                    st.metric("Completeness Violations", violations_count)
                    
            else:
                st.success(f"No completeness check details found for {display_brand}.")
                st.info("This could mean either all checks passed or no completeness rules are configured for this brand.")
                
        except Exception as e:
            st.error(f"Could not load data from DQ_DETAILS_COMPLETENESS: {e}")
            st.write("This table may not exist yet. Completeness check details will appear here after rules are executed.")

    generate_ai_insights_for_tab(
        data_df=completeness_pd_df,
        analysis_type="Completeness Check Results",
        brand_name=display_brand,
        relevant_cols=['EXPECTED_DATE', 'ACTUAL_MAX_DATE', 'DAYS_MISSING', 'EXECUTION_TIMESTAMP']
    )

# --- TAB 4: Historical Completeness Details ---
with tab4:
    st.header("Historical Completeness Details")
    st.markdown("This section shows historical completeness checks that monitor record count consistency over 60-day rolling windows.")
    
    historical_completeness_pd_df = pd.DataFrame()
    with st.spinner("Loading historical completeness details..."):
        try:
            # First check if we have any HISTORICAL_COMPLETENESS rules for this brand in the summary
            all_summary_df = session.table("DQ_RESULTS").to_pandas()
            brand_summary_df = filter_df_by_brand(all_summary_df, data_brand)
            hist_completeness_rules = brand_summary_df[brand_summary_df['RULE_TYPE'] == 'HISTORICAL_COMPLETENESS']
            
            st.subheader("Historical Completeness Rules Summary")
            if not hist_completeness_rules.empty:
                st.write(f"Found {len(hist_completeness_rules)} Historical Completeness rule(s) for {display_brand}:")
                st.dataframe(hist_completeness_rules[['RULE_NAME', 'RULE_TYPE', 'INDICATOR', 'RESULT_VALUE', 'EXECUTION_TIMESTAMP']])
            else:
                st.info(f"No Historical Completeness rules found for {display_brand} in the DQ_RESULTS table.")
            
            st.divider()
            st.subheader("Detailed Historical Completeness Results")
            
            # Load the detailed historical completeness records
            try:
                all_hist_completeness_details = session.table("DQ_HISTORICAL_COMPLETENESS_DETAILS").to_pandas()
                
                # Debug: Show what columns are available
                if st.checkbox("Show table structure for debugging", key="debug_hist_completeness"):
                    st.write("Available columns in DQ_HISTORICAL_COMPLETENESS_DETAILS:")
                    st.write(list(all_hist_completeness_details.columns))
                    st.write("Sample data:")
                    st.dataframe(all_hist_completeness_details.head())
                
                # Try to filter, but handle cases where expected columns might not exist
                if not all_hist_completeness_details.empty:
                    # Check if this table has the columns needed for brand filtering
                    if 'RULE_NAME' in all_hist_completeness_details.columns:
                        historical_completeness_pd_df = filter_df_by_brand(all_hist_completeness_details, data_brand)
                    else:
                        # If no RULE_NAME column, try to filter by other brand-related columns
                        if 'DATASET_NAME' in all_hist_completeness_details.columns and data_brand != "__GLOBAL__":
                            # Filter by dataset name containing the brand
                            historical_completeness_pd_df = all_hist_completeness_details[
                                all_hist_completeness_details['DATASET_NAME'].str.contains(data_brand, case=False, na=False)
                            ]
                        else:
                            # Show all data if no filtering possible
                            historical_completeness_pd_df = all_hist_completeness_details
                else:
                    historical_completeness_pd_df = pd.DataFrame()
                    
            except Exception as detail_error:
                st.error(f"Could not load historical completeness data: {detail_error}")
                st.error(f"Available tables and columns for debugging:")
                # Show available columns in the table
                try:
                    table_info = session.sql("DESCRIBE TABLE DQ_HISTORICAL_COMPLETENESS_DETAILS").to_pandas()
                    st.write("DQ_HISTORICAL_COMPLETENESS_DETAILS columns:")
                    st.dataframe(table_info)
                except Exception as describe_error:
                    st.error(f"Could not describe table: {describe_error}")
                historical_completeness_pd_df = pd.DataFrame()
            
            if not historical_completeness_pd_df.empty:
                st.write(f"Found {len(historical_completeness_pd_df)} historical completeness detail(s):")
                
                # Check for data lag by comparing DATA_DATE with expected current date
                if 'DATA_DATE' in historical_completeness_pd_df.columns:
                    # Get the most recent data date and compare with expected (yesterday)
                    from datetime import datetime, timedelta
                    today = datetime.now().date()
                    expected_date = today - timedelta(days=1)
                    
                    # Convert DATA_DATE to date for comparison
                    historical_completeness_pd_df['DATA_DATE'] = pd.to_datetime(historical_completeness_pd_df['DATA_DATE']).dt.date
                    most_recent_data_date = historical_completeness_pd_df['DATA_DATE'].max()
                    
                    if most_recent_data_date < expected_date:
                        days_behind = (expected_date - most_recent_data_date).days
                        st.warning(f"⚠️ **Data Freshness Notice:**")
                        st.warning(f"📅 Data is {days_behind} day(s) behind expected date. Using {most_recent_data_date} instead of {expected_date}.")
                        st.info(f"💡 The Historical Completeness rule automatically adjusts for data lag by using the most recent available date for calculations.")
                        st.divider()
                
                # Add metrics for better visualization
                if 'STATUS' in historical_completeness_pd_df.columns:
                    pass_count = len(historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'PASS'])
                    fail_count = len(historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'FAIL'])
                    error_count = len(historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'ERROR'])
                    
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Passed Checks", f"{pass_count}")
                    metric_cols[1].metric("Failed Checks", f"{fail_count}", delta_color="inverse")
                    metric_cols[2].metric("Error Checks", f"{error_count}", delta_color="inverse")
                    
                    # Show tolerance threshold info if available
                    if 'TOLERANCE_THRESHOLD' in historical_completeness_pd_df.columns and not historical_completeness_pd_df.empty:
                        tolerance_threshold = historical_completeness_pd_df['TOLERANCE_THRESHOLD'].iloc[0]
                        metric_cols[3].metric("Tolerance Threshold", f"{tolerance_threshold:.1%}", 
                                            help="Deviation percentage above this threshold triggers a failure")
                
                # Show error checks prominently
                error_checks = historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'ERROR']
                if not error_checks.empty:
                    st.subheader("⚠️ Error Historical Completeness Checks")
                    st.error(f"Found {len(error_checks)} dataset(s) with processing errors. This may indicate data access issues or missing data.")
                    st.dataframe(error_checks)
                    st.divider()
                
                # Show failed checks prominently
                failed_checks = historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'FAIL']
                if not failed_checks.empty:
                    st.subheader("🚨 Failed Historical Completeness Checks")
                    st.dataframe(failed_checks)
                    st.divider()
                
                # Show all details
                st.subheader("All Historical Completeness Results")
                st.dataframe(historical_completeness_pd_df)
                
                # Show summary statistics if available
                if 'DEVIATION_PERCENTAGE' in historical_completeness_pd_df.columns:
                    st.subheader("Deviation Statistics")
                    numeric_data = historical_completeness_pd_df[historical_completeness_pd_df['DEVIATION_PERCENTAGE'].notna()]
                    if not numeric_data.empty:
                        avg_deviation = numeric_data['DEVIATION_PERCENTAGE'].mean()
                        max_deviation = numeric_data['DEVIATION_PERCENTAGE'].max()
                        min_deviation = numeric_data['DEVIATION_PERCENTAGE'].min()
                        
                        stat_cols = st.columns(4)
                        stat_cols[0].metric("Average Deviation", f"{avg_deviation:.4f}%")
                        stat_cols[1].metric("Maximum Deviation", f"{max_deviation:.4f}%")
                        stat_cols[2].metric("Minimum Deviation", f"{min_deviation:.4f}%")
                        
                        # Add rolling window analysis summary
                        if 'TOLERANCE_THRESHOLD' in historical_completeness_pd_df.columns:
                            tolerance_exceeded = len(historical_completeness_pd_df[
                                historical_completeness_pd_df['DEVIATION_PERCENTAGE'] > 
                                historical_completeness_pd_df['TOLERANCE_THRESHOLD']
                            ])
                            stat_cols[3].metric("Tolerance Violations", f"{tolerance_exceeded}", 
                                              delta_color="inverse" if tolerance_exceeded > 0 else "normal")
                
                # Enhanced audit trail information if available
                if not historical_completeness_pd_df.empty and 'DROPPED_DAY_DATE' in historical_completeness_pd_df.columns:
                    st.subheader("Rolling Window Audit Trail")
                    
                    # Show audit columns in a more organized way
                    audit_cols = ['DATASET_NAME', 'DATA_DATE', 'DROPPED_DAY_DATE', 'DROPPED_DAY_COUNT', 
                                'NEWEST_DAY_COUNT', 'PREVIOUS_CUMULATIVE_COUNT', 'ACTUAL_60_DAY_COUNT', 
                                'EXPECTED_60_DAY_COUNT', 'DEVIATION']
                    available_audit_cols = [col for col in audit_cols if col in historical_completeness_pd_df.columns]
                    
                    if available_audit_cols:
                        st.write("**Rolling Window Calculation Details:**")
                        st.dataframe(historical_completeness_pd_df[available_audit_cols], use_container_width=True)
                        
                        # Explain the calculation
                        st.info("💡 **Rolling Window Logic**: Expected Count = Previous 59-Day Total - Dropped Day (Day 61) + Newest Day. This ensures consistent 60-day window monitoring.")
            else:
                st.success(f"No historical completeness issues found for {display_brand} in DQ_HISTORICAL_COMPLETENESS_DETAILS table.")
                
        except Exception as e:
            st.error(f"Could not load historical completeness data: {e}")
            st.write("Available tables and columns for debugging:")
            try:
                # Show available tables for debugging
                tables_df = session.sql("SHOW TABLES LIKE 'DQ%'").to_pandas()
                st.write("Available DQ tables:")
                st.dataframe(tables_df)
            except Exception as table_err:
                st.error(f"Could not show tables: {table_err}")
    
    # AI Summary for Historical Completeness
    generate_ai_insights_for_tab(
        data_df=historical_completeness_pd_df[historical_completeness_pd_df['STATUS'] == 'FAIL'] if not historical_completeness_pd_df.empty and 'STATUS' in historical_completeness_pd_df.columns else pd.DataFrame(),
        analysis_type="Historical Completeness",
        brand_name=display_brand,
        relevant_cols=['DATASET_NAME', 'DATA_DATE', 'ACTUAL_60_DAY_COUNT', 'EXPECTED_60_DAY_COUNT', 'DEVIATION', 'DEVIATION_PERCENTAGE', 'STATUS', 'TOLERANCE_THRESHOLD'],
        full_data_df=historical_completeness_pd_df
    )

# --- TAB 5: Sustained Trend Details ---
with tab5:
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
            
    generate_ai_insights_for_sustained_trends(
        data_df=details_pd_df,
        brand_name=display_brand
    )

# --- TAB 6: Missing Data Details ---
with tab6:
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

    generate_ai_insights_for_tab(
        data_df=missing_data_pd_df,
        analysis_type="Missing Data and Null Violations",
        brand_name=display_brand,
        relevant_cols=['FAILURE_TYPE', 'FAILURE_DATE', 'DETAILS']
    )

# --- TAB 7: Spike / Dip Details ---
with tab7:
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

    generate_ai_insights_for_spike_dip(
        data_df=spike_dip_brand_df,
        brand_name=display_brand
    )

# --- TAB 8: Sigma & Anomaly Analysis (MERGED) ---
with tab8:
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

        generate_ai_insights_for_tab(data_df=sigma_outliers, analysis_type="3 Sigma Outlier Analysis", brand_name=display_brand, relevant_cols=['ORDER_DATE', 'Actual Orders', '+3 Sigma', '-3 Sigma', 'Zone'])
        
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

        generate_ai_insights_for_tab(data_df=anomalies_pd_df, analysis_type="2-to-3 Sigma Warning Analysis", brand_name=display_brand, relevant_cols=['ORDER_DATE', 'Actual Orders', '60-Day Rolling Average', 'Zone'])

# --- TAB 9: Negative Value Details ---
with tab9:
    st.subheader(f"🚨 Negative Value Violations - {display_brand}")
    st.write("This tab shows records where numeric fields contain negative values, which may indicate data quality issues.")
    
    with st.spinner(f"Loading negative value violations for {display_brand}..."):
        try:
            # Query negative value details
            negative_details_df = session.table("DQ_NEGATIVE_VALUE_DETAILS").to_pandas()
            
            if negative_details_df.empty:
                st.success("✅ No negative value violations found!")
                st.balloons()
            else:
                # Apply brand filtering if needed
                if display_brand != "All Brands":
                    # Extract brand from RECORD_ID format: YYYY-MM-DD-BRAND-PLATFORM
                    def extract_brand_from_record_id(record_id):
                        try:
                            parts = str(record_id).split('-')
                            if len(parts) >= 5:
                                # Format: YYYY-MM-DD-BRAND-PLATFORM
                                return parts[3]
                            elif len(parts) >= 3:
                                # Fallback: assume simple format DATE-BRAND-PLATFORM
                                return parts[1]
                            return None
                        except:
                            return None
                    
                    negative_details_df['EXTRACTED_BRAND'] = negative_details_df['RECORD_ID'].apply(extract_brand_from_record_id)
                    
                    # Handle brand name variations and filtering
                    def normalize_brand_name(brand_name):
                        """Normalize brand names for comparison"""
                        if not brand_name:
                            return ""
                        brand_name = str(brand_name).upper().strip()
                        # Handle common variations
                        if "BARRON" in brand_name:
                            return "BARRONS"
                        elif "MARKETWATCH" in brand_name or "MARKET WATCH" in brand_name:
                            return "MARKETWATCH"
                        elif "WSJ" in brand_name or "WALL STREET" in brand_name:
                            return "WSJ"
                        return brand_name
                    
                    # Normalize both the display brand and extracted brands
                    normalized_display_brand = normalize_brand_name(display_brand)
                    negative_details_df['NORMALIZED_BRAND'] = negative_details_df['EXTRACTED_BRAND'].apply(normalize_brand_name)
                    
                    # Filter based on normalized brand names
                    if display_brand == "__GLOBAL__":
                        # For Global/Uncategorized, show records where brand extraction failed or is unknown
                        filtered_negative_df = negative_details_df[
                            (negative_details_df['EXTRACTED_BRAND'].isna()) | 
                            (negative_details_df['EXTRACTED_BRAND'] == '') |
                            (negative_details_df['NORMALIZED_BRAND'] == '')
                        ]
                    else:
                        filtered_negative_df = negative_details_df[
                            negative_details_df['NORMALIZED_BRAND'] == normalized_display_brand
                        ]
                else:
                    filtered_negative_df = negative_details_df
                
                if filtered_negative_df.empty:
                    st.success(f"✅ No negative value violations found for {display_brand}!")
                else:
                    st.error(f"❌ Found {len(filtered_negative_df)} negative value violations for {display_brand}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Violations", len(filtered_negative_df))
                    with col2:
                        unique_fields = filtered_negative_df['FIELD_NAME'].nunique()
                        st.metric("Affected Fields", unique_fields)
                    with col3:
                        unique_records = filtered_negative_df['RECORD_ID'].nunique()
                        st.metric("Affected Records", unique_records)
                    with col4:
                        latest_detection = filtered_negative_df['DETECTION_TIMESTAMP'].max()
                        st.metric("Latest Detection", latest_detection.strftime('%Y-%m-%d %H:%M') if pd.notna(latest_detection) else "N/A")
                    
                    # Violations by field type
                    st.subheader("📊 Violations by Field Type")
                    field_summary = filtered_negative_df.groupby('FIELD_NAME').agg({
                        'NEGATIVE_VALUE': ['count', 'min', 'max', 'mean'],
                        'RECORD_ID': 'nunique'
                    }).round(2)
                    field_summary.columns = ['Count', 'Min Value', 'Max Value', 'Avg Value', 'Unique Records']
                    st.dataframe(field_summary, use_container_width=True)
                    
                    # Visualization: Violations over time
                    if 'DETECTION_TIMESTAMP' in filtered_negative_df.columns:
                        st.subheader("📈 Violations Detection Timeline")
                        try:
                            # Convert timestamp and create daily summary
                            filtered_negative_df['DETECTION_DATE'] = pd.to_datetime(filtered_negative_df['DETECTION_TIMESTAMP']).dt.date
                            daily_violations = filtered_negative_df.groupby(['DETECTION_DATE', 'FIELD_NAME']).size().reset_index(name='Violation_Count')
                            
                            # Create timeline chart
                            timeline_chart = alt.Chart(daily_violations).mark_bar().encode(
                                x=alt.X('DETECTION_DATE:T', title='Detection Date'),
                                y=alt.Y('Violation_Count:Q', title='Number of Violations'),
                                color=alt.Color('FIELD_NAME:N', title='Field Name'),
                                tooltip=['DETECTION_DATE:T', 'FIELD_NAME:N', 'Violation_Count:Q']
                            ).properties(
                                width=700,
                                height=400,
                                title=f"Daily Negative Value Violations - {display_brand}"
                            )
                            st.altair_chart(timeline_chart, use_container_width=True)
                        except Exception as chart_error:
                            st.warning(f"Could not create timeline chart: {chart_error}")
                    
                    # Distribution of negative values
                    st.subheader("📉 Distribution of Negative Values")
                    for field in filtered_negative_df['FIELD_NAME'].unique():
                        field_data = filtered_negative_df[filtered_negative_df['FIELD_NAME'] == field]
                        
                        with st.expander(f"📋 {field} Details ({len(field_data)} violations)"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write("**Statistics:**")
                                st.write(f"• Count: {len(field_data)}")
                                st.write(f"• Min: {field_data['NEGATIVE_VALUE'].min()}")
                                st.write(f"• Max: {field_data['NEGATIVE_VALUE'].max()}")
                                st.write(f"• Average: {field_data['NEGATIVE_VALUE'].mean():.2f}")
                            
                            with col2:
                                # Histogram of negative values
                                try:
                                    hist_chart = alt.Chart(field_data).mark_bar().encode(
                                        x=alt.X('NEGATIVE_VALUE:Q', bin=True, title='Negative Value'),
                                        y=alt.Y('count()', title='Frequency'),
                                        tooltip=['count()', 'NEGATIVE_VALUE:Q']
                                    ).properties(
                                        width=400,
                                        height=200,
                                        title=f"Distribution of {field} Negative Values"
                                    )
                                    st.altair_chart(hist_chart, use_container_width=True)
                                except Exception as hist_error:
                                    st.warning(f"Could not create histogram for {field}: {hist_error}")
                    
                    # Detailed records table
                    st.subheader("📋 Detailed Violation Records")
                    
                    # Add filters
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_fields = st.multiselect(
                            "Filter by Field:", 
                            options=filtered_negative_df['FIELD_NAME'].unique(),
                            default=filtered_negative_df['FIELD_NAME'].unique(),
                            key="negative_field_filter"
                        )
                    with col2:
                        min_value = st.number_input(
                            "Minimum Negative Value:", 
                            value=float(filtered_negative_df['NEGATIVE_VALUE'].min()),
                            key="min_negative_value"
                        )
                    
                    # Apply filters
                    display_df = filtered_negative_df[
                        (filtered_negative_df['FIELD_NAME'].isin(selected_fields)) &
                        (filtered_negative_df['NEGATIVE_VALUE'] >= min_value)
                    ].sort_values('DETECTION_TIMESTAMP', ascending=False)
                    
                    # Display the filtered table
                    st.dataframe(
                        display_df[['RECORD_ID', 'FIELD_NAME', 'NEGATIVE_VALUE', 'RECORD_TIMESTAMP', 'DETECTION_TIMESTAMP']],
                        use_container_width=True,
                        column_config={
                            'RECORD_ID': 'Record ID',
                            'FIELD_NAME': 'Field',
                            'NEGATIVE_VALUE': st.column_config.NumberColumn('Negative Value', format="%.2f"),
                            'RECORD_TIMESTAMP': 'Record Date',
                            'DETECTION_TIMESTAMP': 'Detected At'
                        }
                    )
                    
                    # Export functionality
                    if st.button("📥 Download Violation Details", key="download_negative_values"):
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"negative_value_violations_{display_brand}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_negative_csv"
                        )
                        
        except Exception as e:
            st.error(f"An error occurred while loading negative value violations: {e}")
    
    # Generate AI insights for negative values
    if 'filtered_negative_df' in locals() and not filtered_negative_df.empty:
        generate_ai_insights_for_tab(
            data_df=filtered_negative_df, 
            analysis_type="Negative Value Analysis", 
            brand_name=display_brand, 
            relevant_cols=['FIELD_NAME', 'NEGATIVE_VALUE', 'RECORD_ID', 'DETECTION_TIMESTAMP']
        )

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