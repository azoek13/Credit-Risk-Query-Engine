"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated track only) -> escalate OR execute_query
    -> generate_response

Required files next to app.py:
    credit_risk_portfolio.db   (SQLite database, opened read-only)
    test_queries.csv           (optional: sample questions + Evaluation tab)

Credentials (first one found wins):
    1. Streamlit secrets  -> OPENAI_API_KEY, optional OPENAI_API_BASE
    2. Environment vars   -> OPENAI_API_KEY, optional OPENAI_API_BASE / OPENAI_BASE_URL
    3. config.json        -> {"OPENAI_API_KEY": "...", "OPENAI_API_BASE": "..."}  (local use only)
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
# Type hints and structured data types
from typing import TypedDict, List, Any, Dict, Optional
# Data manipulation and analysis
import pandas as pd
# Connect to and query the SQLite database
import sqlite3
# Parse and format SQL queries
import sqlparse
# Read and write JSON data
import json
# Perform regular expression-based text processing
import re
# Interact with the operating system and environment variables
import os
# Capture the pipeline's verbose trace for display in the UI
import io
import contextlib
from datetime import datetime

import streamlit as st

# Use OpenAI chat models through LangChain
from langchain_openai import ChatOpenAI
# Create reusable structured prompts
from langchain_core.prompts import ChatPromptTemplate
# Define system and user messages for LLM calls
from langchain_core.messages import HumanMessage, SystemMessage

# Manage Python warning messages
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'credit_risk_portfolio.db')
TEST_CSV_PATH = os.path.join(BASE_DIR, 'test_queries.csv')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')


# ------------------------------------------------------------------
# OpenAI credentials (Streamlit secrets -> env vars -> config.json)
# ------------------------------------------------------------------
def load_credentials():
    api_key, api_base = None, None

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3. config.json (same format as the notebook)
    if (not api_key) and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as file:
            config = json.load(file)
            api_key = config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Add `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`) "
        "to Streamlit secrets, set it as an environment variable, or provide a `config.json`."
    )
    st.stop()

# Store API credentials in environment variables (as in the notebook)
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# ------------------------------------------------------------------
# Model initialization
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_llms():
    # lightweight model for primary reasoning, classification, and generation
    _llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    # capable model for validation and evaluation, ensuring better accuracy, reliability, and cost efficiency
    _evaluator_llm = ChatOpenAI(model='gpt-4o', temperature=0)
    return _llm, _evaluator_llm


llm, evaluator_llm = init_llms()


# ------------------------------------------------------------------
# Database loading (read-only connection)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(path):
    # read only mode connection; check_same_thread=False because Streamlit runs scripts in worker threads
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)


if not os.path.exists(DB_PATH):
    st.error(f"Database file not found: `{DB_PATH}`. Place `credit_risk_portfolio.db` next to app.py.")
    st.stop()

conn = get_connection(DB_PATH)


# ------------------------------------------------------------------
# Database schema provided to the LLM in the prompts
# ------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ------------------------------------------------------------------
# Ground truth / test queries (optional file)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


ground_truth = load_ground_truth(TEST_CSV_PATH)


# ------------------------------------------------------------------
# Verified Query Template Library
# ------------------------------------------------------------------
# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1e6, 2) AS total_outstanding_millions,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss')
                      THEN l.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_exposure_millions
FROM loan_master l
JOIN sector_master s USING (sector_code)
GROUP BY s.sector_name
ORDER BY SUM(l.total_outstanding) DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT loan_category,
ROUND(SUM(total_outstanding)/1e6,2) AS total_outstanding_millions,
COUNT(loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY SUM(total_outstanding) DESC
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT ifrs9_stage,
COUNT(loan_account_number),
ROUND(SUM(ead_amount)/1e6, 2) as ead_amount_millions,
ROUND(SUM(ecl_amount)/1e6, 2) as ecl_amount_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT sector_code,
ROUND(AVG(provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning
JOIN loan_master using (loan_account_number)
JOIN sector_master using (sector_code)
WHERE reporting_date = '2025-09-30'
GROUP BY sector_code
ORDER BY avg_coverage_ratio DESC
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT borrower_name,
sector_code,
ROUND(total_outstanding/1e6, 2) AS total_outstanding_millions,
asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT group_name,
COUNT(loan_account_number) AS loan_count,
ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT loan_account_number,
borrower_name,
sector_code,
ROUND(total_outstanding/1e6, 2) AS total_outstanding_millions,
days_past_due,
asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
CASE
    WHEN days_past_due = 0 THEN '0 (Current)'
    WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
    WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
    WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
    ELSE '90+'
END AS days_past_due,
COUNT(loan_account_number) AS loan_count,
ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY days_past_due
ORDER BY days_past_due DESC
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT borrower_id,
previous_rating,
internal_rating,
pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT reporting_date,
ROUND(SUM(ecl_amount)/1e6, 2) AS ecl_amount_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date DESC
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ------------------------------------------------------------------
# Tool: Intent Classification
# ------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a credit risk analyst for a bank. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'exposure' or 'outstanding' means total_outstanding; 'NPA', 'bad loans' or 'stressed assets' means asset_classification of Substandard, Doubtful or Loss; 'overdue' or 'past due' means delinquent accounts; 'industry' means sector
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. If the user asks for a threshold, borrower, sector, state, product type, or date range the template does not support, use the generated route. A borrower-level or loan-level question must not match an aggregate template, and an aggregate question must not match a row-level template.
7. Be careful about shape of answer: a question asking for row-level detail should NOT match a template that returns an aggregate count.


### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ------------------------------------------------------------------
# Tool: Query Generation
# ------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

###ROLE
You are a senior SQL developer specializing in credit risk analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Don't use joins unless instructed.
5. Join on the documented keys: loan_master.sector_code to sector_master, loan_master.borrower_id to borrower_rating, and provisioning.loan_account_number to loan_master.
6. Ensure the query is SQLite compatible.
7. Use SQLite-compatible syntax, and julianday() for date differences. Resolve names with sector_name, borrower_name, branch_name or state, not codes.
8. Alias numeric outputs with unit suffixes (_usd, _millions, _pct, _count, _days) and return only the columns needed to answer the question. Add ORDER BY for rankings, and LIMIT for top-N questions.

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ------------------------------------------------------------------
# Tool: Query Validation (five checks)
# ------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about credit risk analysis.

### CONTEXT
{track_context}

### INPUT
User Question:
{user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ------------------------------------------------------------------
# Tool: Retry Generation
# ------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""

### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ------------------------------------------------------------------
# Tool: Query Execution
# ------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ------------------------------------------------------------------
# Tool: Response Generation
# ------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### ROLE
ou are a credit risk analyst at a commercial bank, writing a concise, accurate answer to a portfolio question using only the query result provided, stating exact figures with their units and flagging notable risk concentrations.

### INPUT
User Question:
{user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly in the first sentence.
2. Provide context from other rows only when it adds value (for example, ranking or comparison).
3. Use only figures from the query result, stated exactly and with units (_usd, _millions, _pct, _count, _days). Never estimate or add outside data.
4. State exact numbers from the data. Do not round beyond what is shown.
5. Highlight only the relevant rows instead of repeating the table, and compare figures where it helps (top vs. rest, sector vs. portfolio).
6. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ------------------------------------------------------------------
# Pipeline orchestration
# ------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ------------------------------------------------------------------
# Streamlit helpers
# ------------------------------------------------------------------
def run_and_capture(question):
    """Run the pipeline and capture its verbose trace (the notebook's print output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    result['trace'] = buffer.getvalue()
    result['timestamp'] = datetime.now().isoformat(timespec='seconds')
    return result


def audit_record(result):
    """Serializable audit-trail record for one pipeline run."""
    record = dict(result['log'])
    record['timestamp'] = result.get('timestamp')
    return record


def render_result(result):
    """Render the full trace for one pipeline result."""
    confidence = result['confidence']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Route", str(result['route']).capitalize())
    col2.metric("Query ID", result['query_id'] if result['query_id'] else "—")
    if isinstance(confidence, (int, float)):
        col3.metric("Confidence", f"{confidence:.2f}")
    else:
        col3.metric("Confidence", str(confidence) if confidence is not None else "—")
    col4.metric("Rows Returned", result['row_count'] if result['row_count'] is not None else "—")

    if result['escalated']:
        st.error("Escalated to a human analyst. This question could not be reliably resolved.")
        st.write(result['narrative'])
    else:
        st.subheader("Answer")
        st.write(result['narrative'])

    if result['retry_used']:
        st.info("The first generated query failed validation; one retry was used.")

    st.subheader("SQL")
    if result['executed_sql']:
        st.caption("Executed SQL")
        st.code(sqlparse.format(result['executed_sql'], reindent=True, keyword_case='upper'), language='sql')
    elif result['candidate_sql']:
        st.caption("Candidate SQL (failed validation, not executed)")
        st.code(sqlparse.format(result['candidate_sql'], reindent=True, keyword_case='upper'), language='sql')

    if result['dataframe'] is not None:
        st.subheader("Result data")
        st.dataframe(result['dataframe'], use_container_width=True)
        st.download_button(
            "Download result as CSV",
            result['dataframe'].to_csv(index=False).encode('utf-8'),
            file_name=f"{result['query_id'] or 'generated'}_result.csv",
            mime="text/csv",
            key=f"dl_{result['timestamp']}_{result['user_question'][:20]}",
        )

    with st.expander("Pipeline trace"):
        st.code(result['trace'] or "(no trace)", language='text')
    with st.expander("Validation gate result"):
        st.json(result['gate_result'])
    with st.expander("Audit log record"):
        st.json(audit_record(result))


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Ask routine commercial lending portfolio questions in plain English. "
    "Answers come from pre-approved verified queries where possible, otherwise from validated, "
    "read-only generated SQL. Every answer shows its SQL, raw data and confidence score."
)

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'question_text' not in st.session_state:
    st.session_state['question_text'] = ""

# ---- Sidebar ----
with st.sidebar:
    st.header("About")
    st.write(
        "Read-only PoC for the Northbridge Bank commercial lending portfolio. "
        "Complex or ambiguous questions that fail validation are escalated to a human analyst."
    )
    st.success("Database connected (read-only mode)")

    with st.expander("Verified query library", expanded=False):
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}**: {entry['description']}")

    with st.expander("Database schema", expanded=False):
        st.code(database_schema, language='text')

    if ground_truth is not None:
        st.subheader("Sample questions")

        def _use_sample():
            choice = st.session_state.get('sample_choice')
            if choice and choice != "(choose a sample)":
                st.session_state['question_text'] = choice

        st.selectbox(
            "Test-case questions",
            ["(choose a sample)"] + ground_truth['User Query'].tolist(),
            key='sample_choice',
            on_change=_use_sample,
        )

    if st.button("Clear session history"):
        st.session_state['history'] = []
        st.rerun()

tab_ask, tab_history, tab_eval = st.tabs(["Ask a question", "Audit trail (this session)", "Evaluation"])

# ---- Ask tab ----
with tab_ask:
    question = st.text_area(
        "Your question",
        key='question_text',
        height=100,
        placeholder="e.g. How much of our book is in real estate, and how much of it is NPA?",
    )

    if st.button("Run query", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running pipeline: classify → build SQL → validate → execute → respond..."):
                try:
                    result = run_and_capture(question.strip())
                    st.session_state['history'].append(result)
                    st.session_state['latest'] = result
                except Exception as e:
                    st.session_state.pop('latest', None)
                    st.error(f"Pipeline error: {e}")

    if st.session_state.get('latest') is not None:
        render_result(st.session_state['latest'])

# ---- Audit trail tab ----
with tab_history:
    history = st.session_state['history']
    if not history:
        st.info("No queries run yet in this session.")
    else:
        audit_df = pd.DataFrame([
            {
                'Time': r.get('timestamp'),
                'Question': r['user_question'],
                'Route': r['route'],
                'Query ID': r['query_id'],
                'Retry Used': r['retry_used'],
                'Escalated': r['escalated'],
                'Confidence': r['confidence'],
                'Rows': r['row_count'],
            }
            for r in history
        ])
        st.dataframe(audit_df, use_container_width=True)
        st.download_button(
            "Download full audit log (JSON)",
            json.dumps([audit_record(r) for r in history], indent=2, default=str).encode('utf-8'),
            file_name="audit_log.json",
            mime="application/json",
        )

# ---- Evaluation tab ----
with tab_eval:
    st.write(
        "Runs every case in `test_queries.csv` through the pipeline and compares the selected route "
        "and query ID against the ground truth."
    )
    if ground_truth is None:
        st.info("`test_queries.csv` not found next to app.py, so evaluation is unavailable.")
    else:
        st.dataframe(ground_truth, use_container_width=True)
        if st.button("Run evaluation on all test cases"):
            test_results = []
            progress = st.progress(0.0)
            try:
                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    with st.spinner(f"Running {gt['Test Case']} ..."):
                        test_results.append(run_and_capture(gt['User Query']))
                    progress.progress((i + 1) / len(ground_truth))

                evaluation_rows = []
                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    tr = test_results[i]

                    evaluation_rows.append({
                        'Test Case': gt['Test Case'],
                        'Expected Route': gt['Expected Route'],
                        'Actual Route': tr['route'],
                        'Route Match': tr['route'] == gt['Expected Route'],
                        'Expected Query ID': gt['Expected Query ID'],
                        'Actual Query ID': tr['query_id'],
                        'Query ID Match': (
                            pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                        ) or tr['query_id'] == gt['Expected Query ID'],
                        'Confidence': tr['confidence'],
                        'Rows Returned': tr['row_count']
                    })

                evaluation_df = pd.DataFrame(evaluation_rows)

                path_accuracy = evaluation_df['Route Match'].mean() * 100

                verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
                query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

                average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
                m3.metric("Average Confidence Score", f"{average_confidence:.2f}")

                st.dataframe(evaluation_df, use_container_width=True)

                for tr in test_results:
                    st.session_state['history'].append(tr)
            except Exception as e:
                st.error(f"Evaluation error: {e}")
