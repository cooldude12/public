import os
import json
from utils import pretty_print_json, exec_sql, print_debug
# pl use compare_schema.py instead of this file

def get_schema_info(database_name):
    query = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{database_name}'
    ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """
    success, result = exec_sql(query)
    if not success:
        raise Exception(f"Failed to execute query: {result}")

    schema_info = {}
    for row in result:
        table = row['TABLE_NAME']
        column = row['COLUMN_NAME']
        data_type = row['DATA_TYPE']
        if table not in schema_info:
            schema_info[table] = {}
        schema_info[table][column] = data_type
    return schema_info


def compare_schemas(schema_dev, schema_prod):
    comparison_result = {}
    all_tables = set(schema_dev.keys()) | set(schema_prod.keys())

    for table in all_tables:
        table_comparison = {}
        dev_columns = schema_dev.get(table, {})
        prod_columns = schema_prod.get(table, {})

        for column, data_type in dev_columns.items():
            if column not in prod_columns:
                table_comparison[column] = "not present in prod"
            elif dev_columns[column] != prod_columns.get(column):
                table_comparison[column] = "data type different in prod"

        for column in prod_columns:
            if column not in dev_columns:
                table_comparison[column] = "not present in dev"

        if table_comparison:
            comparison_result[table] = table_comparison

    return comparison_result

def get_schema_info(database_name):
    query = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{database_name}'
    ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """
    success, result = exec_sql(query)
    if not success:
        raise Exception(f"Failed to execute query: {result}")

    schema_info = {}
    for row in result:
        table = row[0]  # Assuming TABLE_NAME is the first column
        column = row[1]  # Assuming COLUMN_NAME is the second column
        data_type = row[2]  # Assuming DATA_TYPE is the third column
        if table not in schema_info:
            schema_info[table] = {}
        schema_info[table][column] = data_type
    return schema_info

# Set environment variables for DB connection
dev_db_host='triestai-web-stg-db.cbfszrsab1w8.us-east-2.rds.amazonaws.com'
dev_db_user='triestai'
dev_db_password= 'XXXX' # pl replace it with real pwd
prod_db_host='triestai-web-prd-db.cbfszrsab1w8.us-east-2.rds.amazonaws.com'
prod_db_user='triestai'
prod_db_password='XXXX' # pl replace it with real pwd


os.environ['DB_HOST'] = dev_db_host
os.environ['DB_USER'] = dev_db_user
os.environ['DB_PWD'] = dev_db_password
os.environ['DB_NAME'] = 'triestai'

schema_dev = get_schema_info('triestai')

# Set environment variables for prod DB connection
os.environ['DB_HOST'] = prod_db_host
os.environ['DB_USER'] = prod_db_user
os.environ['DB_PWD'] = prod_db_password
os.environ['DB_NAME'] = 'triestai'

schema_prod = get_schema_info('triestai')

comparison_result = compare_schemas(schema_dev, schema_prod)
print(json.dumps(comparison_result, indent=2))

