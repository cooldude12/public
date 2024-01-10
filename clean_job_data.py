import sys
import os
import sys
import csv
import time    
from utils import print_debug, exec_sql, display_output

def delete_records(question_id):

    # Mapping for special pipeline names to table names
    pipeline_to_table_mapping = {
        "get_topic_llm": "ai_topic_llm",
        "similar_question": "ai_question_similarity"
        # Add other mappings here if needed
    }

    # Fetch records
    sql_fetch = f"SELECT pipeline_job_id, pipeline_name FROM ai.ai_pipeline_jobs WHERE question_id={question_id} ORDER BY 1 DESC;"
    status, records = exec_sql(sql_fetch)

    if not status or not records:
        print("No records found or an error occurred.")
        return

    # Display records in a structured format
    print("Fetched Records:")
    print("{:<12} | {}".format("Job ID", "Pipeline Name"))
    print("-" * 30)
    for record in records:
        print(f"{record[0]:<12} | {record[1]}")

    # Get user input for start and end range
    start_range = int(input("Enter start range for pipeline_job_id: "))
    end_range = int(input("Enter end range for pipeline_job_id: "))

    # Construct and execute delete SQL for each record in range
    for record in records:
        job_id, pipeline_name = record
        if start_range <= job_id <= end_range:
            # Map to correct table name if needed
            final_table_name = pipeline_to_table_mapping.get(pipeline_name, f"ai_{pipeline_name}")
            sql_delete_job = f"DELETE FROM ai.ai_pipeline_jobs WHERE pipeline_job_id={job_id};"
            sql_delete_table = f"DELETE FROM ai_pipeline.{final_table_name} WHERE pipeline_job_id={job_id};"
            
            # sql_delete_job = f"DELETE FROM ai.ai_pipeline_jobs WHERE pipeline_job_id={job_id};"
            # sql_delete_table = f"DELETE FROM ai_pipeline.ai_{pipeline_name} WHERE pipeline_job_id={job_id};"

            # print("Executing:", sql_delete_job)
            exec_sql(sql_delete_job)
            print("Executing:", sql_delete_table)
            exec_sql(sql_delete_table)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <question_id>")
        sys.exit(1)

    question_id = sys.argv[1]
    delete_records(question_id)
