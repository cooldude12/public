import datetime
import inspect
import os
import csv
import subprocess
import re
import sys, select
import string
import pytz
import pymysql
import json
import time

DEBUG_ON = True
if os.getenv('ENV') != "PROD":
    DEBUG_ON_DISPLAY = True
else:
    DEBUG_ON_DISPLAY = False
LOAD_DEBUG_DB=False
LOG_TO_FILE = True
DEBUG_FILE_PATH = '~/logs'
OUT_FILE_PATH = '~/out'
LOG_BATCH_REC_COUNT=50
log_batch = []  # Initialize the global log_batch list

def get_current_time_pst(time_type='day'):
    # Get current time using function with default value
    # Set the timezone to PST
    tz = pytz.timezone('US/Pacific')
    # Get the current time in PST
    current_time = datetime.datetime.now(tz)
    # Format the current time based on time_type
    if time_type == 'day':
        formatted_time = current_time.strftime("%Y-%m-%d")
    elif time_type == 'second':
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    else:
        formatted_time = current_time.strftime("%Y-%m-%d-%H")
    return formatted_time

# display output one line at a time
def display_output(arr):
    if arr is None:
        print("Output is not available.")
        return
    
    for row in arr:
        if row is not None:
            print(str(row))

def get_calling_program_name():
    frame = inspect.stack()[-1]
    calling_file = frame[0].f_code.co_filename
    calling_program = calling_file.split('/')[-1].split('.')[0]
    return calling_program

def print_debug(msg):
    if os.getenv('ENV') == "PROD":
        debug_msg = f"{datetime.datetime.now().strftime('%H:%M:%S')} - {msg}\n"
    else:
        debug_msg = f"{msg}\n"
    
    global log_batch # Declare log_batch as a global variable inside the function

    if DEBUG_ON_DISPLAY:
        print(str(debug_msg))
            
    if DEBUG_ON:
        calling_program = get_calling_program_name()
        current_dt = get_current_time_pst('day')
        # Check for PIPELINE_JOB_ID environment variable
        pipeline_job_id = os.environ.get('PIPELINE_JOB_ID', '')
        filename_suffix = f".{pipeline_job_id}" if pipeline_job_id else ""
        log_filename = f"{DEBUG_FILE_PATH}/{calling_program}.{current_dt}{filename_suffix}.log"
        #log_filename = DEBUG_FILE_PATH + "/" + f"{calling_program}.{current_dt}.log"
        log_filename_modified = os.path.expanduser(log_filename)
        if not os.path.exists(log_filename_modified):
            os.makedirs(os.path.dirname(log_filename_modified), exist_ok=True)

        # Open the log file
        mode = 'a' if os.path.exists(log_filename_modified) else 'w'
        with open(log_filename_modified, mode) as f:
            if LOG_TO_FILE:
                f.write(str(debug_msg))
            else:
                print(debug_msg, end='')
        if LOAD_DEBUG_DB:
        # insert this record in the table 
            pipeline_job_id = os.environ.get('PIPELINE_JOB_ID', '')
            log_batch.append((pipeline_job_id, debug_msg))
            #print_debug("batch log lenght = " + str(len(log_batch)) + " threshold = " + str(LOG_BATCH_REC_COUNT))
            if len(log_batch) >= LOG_BATCH_REC_COUNT:
                print("now inserting batched logs in DB")
                batch_insert_logs()
                log_batch = []
                    
def batch_insert_logs():
    insert_query = "INSERT INTO ai.ai_job_logs (pipeline_job_id, log) VALUES (%s, %s)"
    print(log_batch)
    exec_sql(insert_query, log_batch)
    
def exec_sql(sql, values=None):
    #print(os.getenv('DB_HOST') + " " + os.getenv('DB_USER') + " " + os.getenv('DB_PWD') + " " + os.getenv('DB_NAME'))
    connection = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PWD'),
        database=os.getenv('DB_NAME')
    )

    #print_debug(sql)
    try:
        with connection.cursor() as cursor:
            if values is not None:
                print_debug("values = " + str(values))
                cursor.execute(sql, values)
            else:
                cursor.execute(sql)

            connection.commit()
            result = cursor.fetchall()
            #print_debug("after fetching result" + str(result))
            return True, result
    except Exception as e:
        print_debug(sql)
        print_debug(f"\nError executing SQL: {e}")
        print_debug("values = " + str(values))
        return False, str(e)
        sys.exit(1)
    finally:
        connection.close()

global_connection = None
def get_db_connection():
    global global_connection
    if global_connection is None or not global_connection.open:
        global_connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PWD'),
            database=os.getenv('DB_NAME')
        )
    return global_connection

def exec_sql1(sql, values=None):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if values is not None:
                cursor.execute(sql, values)
            else:
                cursor.execute(sql)
            connection.commit()
            result = cursor.fetchall()
            return True, result
    except Exception as e:
        print_debug(sql)
        print_debug(f"\nError executing SQL: {e}")
        print_debug("values = " + str(values))
        return False, str(e)
        sys.exit(1)

# print pretty
def pretty_print_json(json_data):
    pretty_json = json.dumps(json_data, indent=2)
    print("\n----------------------------------")
    print(pretty_json)

def wait_for_user_prompt(index_dict):
    user_prompt_enabled = os.getenv('USER_PROMPT_ENABLED', '').lower() == 'true'
    
    if not user_prompt_enabled:
        index_dict['index'] += 1
        return
    
    print("Press Enter to continue or 0 to exit...")
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            user_input = sys.stdin.readline().strip()
            if user_input == '0':
                sys.exit("Exiting program...")
            index_dict['index'] += 1
            break

def replace_substring_in_nested_list(nested_list, substring, replacement):
    for i, sublist in enumerate(nested_list):
        for j, element in enumerate(sublist):
            if isinstance(element, str):
                nested_list[i][j] = element.replace(substring, replacement)
    return nested_list

def write_sql_out_to_csv(sql_query, fname):
    current_hour = get_current_time_pst()
    status, result = exec_sql(sql_query)
    if result:
        out_filename = os.path.join(OUT_FILE_PATH, fname)
        out_filename_modified = os.path.expanduser(out_filename)
        os.makedirs(os.path.dirname(out_filename_modified), exist_ok=True)

        # Open the log file
        mode = 'a' if os.path.exists(out_filename_modified) else 'w'
        with open(out_filename_modified, mode, newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            for row in result:
                writer.writerow(row)
    else:
        print_debug("No records to write in outfile")

def current_unix_time():
    return int(time.time())

def call_time_dur(end_time, start_time, label="Step"):
    duration = end_time - start_time
    print_debug(f"{label} duration: {duration} sec")

def start_timer():
    return time.time()

def end_timer(start_time, label):
    end_time = time.time()
    duration = end_time - start_time
    print(f"{label} duration : {duration:.2f} sec")
