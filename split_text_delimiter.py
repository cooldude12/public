"""
This function splits input sentences into a list of sentences based on specified delimiters.

Inputs:
- input_data: A list of input sentences to be split, in the format [[question_id, answer_id, feedback], ...].
- delimiters: A list of delimiters to use for splitting the sentences.

Returns:
A list of modified sentences, in the same format as the input data.

Example usage:
input_data = [    [1, 1, "The product is expensive, but the service is good and the packaging is attractive."],
    [1, 2, "I don't like the color, but the design is nice."],
    [2, 1, "The shipping was fast. The product arrived in good condition."]
]

input data has question_id, answer_id, answer_text 
output:  question_id, answer_id, answer_text_modified

delimiters = [",", ".", "but", "and"]
output_data = split_feedback(input_data, delimiters)

"""

import re
import pandas as pd
import re
import sys
from utils import print_debug, exec_sql, display_output, pretty_print_json
import json
    
def main(operation,pipeline_job_id=0, delimiters=""):
    print_debug("in split_feedback main(), operation=" + operation + " pipleine_id=" + str(pipeline_job_id) + " Delmeters:" + str(delimiters))
    if operation == "unit_test":
        reviews, delimiters = unit_test()
        print_debug("unit test : " + str(delimiters))
    elif operation == "split_text":
        reviews = get_data_for_split(pipeline_job_id)
    else:
        print_debug("invalid option")
        sys.exit(1)
    
    output_data = split_text(reviews, delimiters)
    pretty_print_json(output_data)

    if operation == "split_text":
        save_data_db(output_data, pipeline_job_id)
        update_input_data_job_status(pipeline_job_id)
        return True
    else:
        print_debug(output_data)
        return True

def get_data_for_split(pipeline_job_id):
    # Retrieve the search data array
    sql = f"SELECT ai.question_id, ai.answer_id,  \
            JSON_UNQUOTE(JSON_EXTRACT(ai.pipeline_input, '$[0]')) as answer_text \
            FROM ai_pipeline.ai_split_text_delimiter ai \
            WHERE job_status='Scheduled' AND pipeline_job_id={pipeline_job_id}"

    status, result = exec_sql(sql)
    if not status:
        return
    
    data = []
    for row in result:
        answer_id = row[0]
        question_id = row[1]
        answer_text =  clean_text(row[2])
        data.append([question_id, answer_id, answer_text])

    if len(data) == 0:
        print_debug("** WARNING: No answers fetched")
    else:
        print_debug("set of answers to be processed by the ML model:")
        print_debug(data)
        return data
    return


def split_text(list1, list2):
    result = []
    #delimiters = json.loads(list2)
    delimiters = list2
    
    for question_id, answer_id, review in list1:
        # Initialize a list to store split subsets of the review
        split_subsets = [review]

        # Iterate through the delimiters and split the review
        for delimiter in delimiters:
            new_subsets = []
            for subset in split_subsets:
                new_subsets.extend(subset.split(delimiter))
            split_subsets = new_subsets

        # Format subsets with double quotes and remove empty strings
        formatted_subsets = [subset.strip(' ,') for subset in split_subsets if subset.strip() != '']
        print_debug("post split:" + str(formatted_subsets))

        # Append the result for the current review to the output list
        result.append([question_id, answer_id, review, formatted_subsets])

    return result

def update_input_data_job_status(pipeline_job_id):
    sql_ai_user_data = f"UPDATE ai_pipeline.ai_split_text_delimiter SET job_status='Processed' \
          WHERE job_status='Scheduled' and pipeline_job_id={pipeline_job_id}"
    exec_sql(sql_ai_user_data)

def save_data_db(data, pipeline_job_id):
    delete_query = f"DELETE FROM ai.ai_split_text WHERE pipeline_job_id={pipeline_job_id}"
    exec_sql(delete_query)
        
    for row in data:
        answer_id = row[0]
        question_id = row[1]
        answer_text = row[2]
        json_labels = json.dumps(row[3])
        
        insert_query = "INSERT INTO ai.ai_split_text \
                (pipeline_job_id, question_id, answer_id, \
                answer_text, label) \
                VALUES (%s, %s, %s, %s, %s)"
        values = (pipeline_job_id, question_id, answer_id, answer_text, json_labels)
        # delete existing records
        exec_sql(insert_query, values)
    
    # update the CRM ai db with the labels
    update_query_crm = f"""UPDATE ai_pipeline.ai_split_text_delimiter a join ai.ai_split_text b 
                ON a.pipeline_job_id=b.pipeline_job_id 
                AND a.answer_id=b.answer_id 
                -- AND JSON_UNQUOTE(JSON_EXTRACT(a.pipeline_input, '$[0]')) = b.answer_text
                AND a.pipeline_job_id={pipeline_job_id}
                SET a.pipeline_output=b.label,
                a.job_status='Processed'
                """    
    exec_sql(update_query_crm)

# create one row for each sentence post split
# and remove the original sentence
def flatten_output(input_arr):
    output_arr = []
    for row in input_arr:
        row_id = row[0]
        row_subid = row[1]
        row_text = row[2]
        sentences = [sent[0] for sent in row[3]]
        for sentence in sentences:
            output_arr.append([row_id, row_subid, row_text, sentence])
    return output_arr

import re
def clean_text(text):
    # Remove square brackets and double quotes
    cleaned_text = text.replace("[", "").replace("]", "").replace('"', '')
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    # Remove any remaining control characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
    return cleaned_text


def unit_test():
    
    list1 = [
        [2, 1, "I don't like the color , but the design is nice."],
        [3, 1, "I do not like the color ,  but the design is nice."],
        [1, 1, "The product is expensive, but the service is \'good\' and the packaging is attractive."]
    ]
    # Add more review entries here if needed
    list2 = ["and", ".", "but", ","]  # Including comma as delimiter
    
    # assert split_feedback(input_data, delimiters) == expected_output
    return (list1, list2)
    
def unit_test():
    input_data = [
        [1, 1, 'The product is expensive, but the service is \'good\' and the packaging is attractive.'],
        [2, 1, 'I don\'t like the  color, but the design is nice.'],
        [3, 1, 'I do not like the color but the design is nice.'],
        [4, 2, 'The shipping was fast. The product arrived in good condition.']

    ]
    input_data = [
        [2, 1, "I don't like the color | but the design is nice."],
        [3, 1, "I do not like the color |  but the design is nice."],
        [5,1, '1-Few doctors are loyal prescribers of brands like Lantus | it being the first brand from the originators 2- Doctors engaged in activities with Sanofi (Lantus) 3- Free pens are provided in surplus against demand(need)by Lantus and Basalog and there is no space to accomodate a third or fourth brand 4- Marketing executive fail to promote the product with proper scientific knowledge and his fear to discuss the brand and convince the doctor 5- Marketing executive priority is other brands like Gluconorm, Telista,Vobit,Lupisulin and his dependence on the potential doctor is too high as he has to meet his target and incentives in a very tough competition'],
        [4, 1, 'The product is expensive , but the service is \'good\' and the packaging is attractive.'],
        [5,1, '1-Few doctors are loyal prescribers of brands like Lantus | it being the first brand from the originators | 2- Doctors engaged in activities with Sanofi (Lantus) | 3- Free pens are provided in surplus against demand(need)by Lantus and Basalog and there is no space to accomodate a third or fourth brand 4- Marketing executive fail to promote the product with proper scientific knowledge and his fear to discuss the brand and convince the doctor 5- Marketing executive priority is other brands like Gluconorm, Telista,Vobit,Lupisulin and his dependence on the potential doctor is too high as he has to meet his target and incentives in a very tough competition']
    ]
    delimiters = ["and", ".", "but", ","] 
    delimiters = ["|"]
    print("in unit test")
    # assert split_feedback(input_data, delimiters) == expected_output
    # output_arr = main(input_data, str(delimiters))
    return input_data, delimiters


def test_split_feedback_consecutive_delimiters():
    input_data = [
        [1, 1, "The product is expensive, but the service is 'good' and the packaging is attractive."]
    ]
    delimiters = ["and", ".", "but", ","]
    expected_output = [
        [1, 1, [['The product is expensive'], ["the service is 'good'"], ['the packaging is attractive']]]
    ]
    print_debug(main(input_data, delimiters))

# Run main function with sample data
if __name__ == "__main__":
    operation = "unit_test"
    main(operation)
    


