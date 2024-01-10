import re
import sys
import os
import json
from utils import print_debug, exec_sql, display_output, pretty_print_json

# Global variable for the NA list file
NA_LIST_FILE_NAME = 'NA_phrases.txt'
# Default path if DATA_FOLDER_LOCATION is not set
DEFAULT_PATH = '/home/ec2-user/code/triestai-backend/config'
DATA_FOLDER_LOCATION = os.getenv('DATA_FOLDER_LOCATION', DEFAULT_PATH)
NA_LIST_FILE = os.path.join(DATA_FOLDER_LOCATION, NA_LIST_FILE_NAME)

def clean_text(text):
    # Emoji removal pattern
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # dingbats
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub(r'', text)

    # Removing square brackets, double quotes, extra whitespaces, and control characters
    cleaned_text = cleaned_text.replace("[", "").replace("]", "").replace('"', '')
    cleaned_text = ' '.join(cleaned_text.split())
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
    return cleaned_text

def remove_NA(sentence_list, na_list):
    # Convert na_list to lowercase for case-insensitive comparison
    na_list_lower = [item.lower() for item in na_list]

    updated_list = []
    for answer_id, sentence in sentence_list:
        # Clean and lowercase the sentence for comparison, but retain original sentence
        cleaned_sentence = clean_text(sentence)
        if cleaned_sentence.lower() not in na_list_lower:
            updated_list.append([answer_id, cleaned_sentence])  # Append the original sentence
    return updated_list

def load_na_list(na_list_str=None):
    if na_list_str:
        print_debug("list passed, so skipping na list file")
        return [word.strip() for word in na_list_str.split(',')]
    else:
        print("list not passed")
        
        with open(NA_LIST_FILE, 'r') as file:
            return [line.strip() for line in file.readlines()]

def get_data_from_db(pipeline_job_id):
    sql = f"SELECT ai.answer_id, \
            JSON_UNQUOTE(JSON_EXTRACT(ai.pipeline_input, '$[0]')) as answer_text \
            FROM ai_pipeline.ai_remove_na ai \
            WHERE job_status='Scheduled' AND pipeline_job_id={pipeline_job_id}"

    status, result = exec_sql(sql)
    if not status:
        print_debug("Error fetching data")
        return []

    data = [[row[0], clean_text(row[1])] for row in result]
    if len(data) == 0:
        print_debug("** WARNING: No answers fetched")
    else:
        print_debug("Set of answers to be processed:")
        print_debug(data)
    return data

def save_data_to_db(result_data, pipeline_job_id):
    # Iterate over the result data to update each row in the database
    for answer_id, processed_text in result_data:
        # Convert the processed text to JSON format
        pipeline_output = json.dumps([processed_text])  # Assuming each row results in a single output string
        # SQL update statement with placeholders for parameters
        update_sql = "UPDATE ai_pipeline.ai_remove_na SET pipeline_output = %s WHERE pipeline_job_id = %s AND answer_id = %s"
        status, _ = exec_sql(update_sql, (pipeline_output, pipeline_job_id, answer_id))

        if not status:
            print_debug(f"Failed to update row with answer_id: {answer_id}")

def unit_test(na_list=None):
    # Define default NA list if none is provided
    default_na_list = ["NA", "Welcome"]
    # Use the provided na_list or the default
    test_na_list = na_list if na_list is not None else default_na_list
    # Define test sentences
    test_sentences = ["Hello 😊", "NA", "Good morning", "Welcome"]
    test_sentences = ["Hello 😊", "NA", "N/A", "N/AA", "Nothing", "Good morning", "Welcome", "but it was good", "but"]
    test_sentences = [[1, "Hello 😊"], [2, "NA"], [3, "N/A"], [4, "N/AA"], [5, "Nothing"], [6, "Good morning"], [7, "Welcome"], [8, "but it was good"], [9, "but"]]
    # Execute the remove_NA function with test data
    print(test_sentences)
    print()
    print(test_na_list)
    return test_sentences, test_na_list
         
def main(operation, pipeline_job_id, na_list_str=None):
    # Check if the NA list file exists
    if not os.path.exists(NA_LIST_FILE):
        print_debug(f"NA list file does not exist: {NA_LIST_FILE}")
        sys.exit(1)
    else:
        print_debug(f"Found NA list file: {NA_LIST_FILE}")

    print_debug("na_list_str=" + str(na_list_str))
    na_list = load_na_list(na_list_str)

    if operation == "remove_na":
        sentence_list = get_data_from_db(pipeline_job_id)
        
    elif operation == "unit_test":
        sentence_list, na_list = unit_test(na_list)
    else:
        print_debug("invalid choice")
        return

    print_debug("raw sentences")    
    print_debug(sentence_list)
    print_debug("NA phrases") 
    print_debug(na_list)
    result = remove_NA(sentence_list, na_list)
    print_debug("Result") 
    print_debug(result)

    if operation == "remove_na":
        save_data_to_db(result, pipeline_job_id)

def test_clean_text():
    # Test cases with expected results
    test_cases = [
        ("Hello 😊", "Hello "),
        ("Good morning 🌞", "Good morning "),
        ("Welcome! 👋", "Welcome! "),
        ("Text with no emoji", "Text with no emoji"),  # No emoji case
        ("Multiple 😊😊😊 emojis", "Multiple  emojis")
    ]

    for text, expected in test_cases:
        result = clean_text(text)
        print("text = " + text)
        print("result = " + result)
        

    print("All tests passed for clean_text")

if __name__ == "__main__":
    operation_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline_job_id_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    na_list_arg = sys.argv[3] if len(sys.argv) > 3 else None
    if operation_arg == "remove_na" and pipeline_job_id_arg is None:
        print("pipeline_job_id_arg is not passed")
        sys.exit(1)
    main(operation_arg, pipeline_job_id_arg, na_list_arg)

