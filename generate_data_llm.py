"""
This script generates few other pipeline data that leverages data for LLM.
"""

from utils import exec_sql, print_debug, pretty_print_json

"""
    update the ai label and split sentence related stuff
"""
def update_ai_labels (question_id, pipeline_job_id):
    # SQL query to fetch the required data
    sql_update = f"""
        UPDATE ai_pipeline.ai_group_label_insight_llm a 
        JOIN (
            SELECT 
                answer_id,
                split_sentence,
                JSON_ARRAY(concat(master_insight_agg, ' : ' , master_insight)) AS output_json,
                JSON_OBJECT(
                        'split_sentence', split_sentence, 
                        'topic', topic,
                        'group_label', group_label_auto, 
                        'insight', insight, 
                        'master_insight', master_insight,
                        'master_insight_agg', master_insight_agg
                    ) as output_json_details
            FROM 
                ai.ai_prep_data_llm_final
            WHERE question_id = {question_id} 
            AND split_sentence IS NOT NULL
        ) b ON JSON_UNQUOTE(JSON_EXTRACT(pipeline_input, '$[0]'))  = b.split_sentence 
        AND b.answer_id = a.answer_id 
        SET 
            a.pipeline_output = b.output_json,
            a.pipeline_output_details = b.output_json_details,
            a.job_status='Processed'
        WHERE 
            a.pipeline_job_id =  {pipeline_job_id}
        """
    print_debug("update ai labels sql start")
    print_debug(sql_update)
    print_debug(" sql end")
    exec_sql(sql_update)

def update_split_sentence (question_id, pipeline_job_id):
    # SQL query to fetch the required data
    sql_update = f"""
    UPDATE ai_pipeline.ai_split_text_llm a 
    JOIN (
        SELECT 
            answer_id,
            ans_txt,
            JSON_ARRAYAGG(split_sentence) AS output_json,
            MAX(full_json_answer_id) as full_json_answer_id
        FROM 
            ai.ai_prep_data_llm_final
        WHERE question_id = {question_id} AND split_sentence IS NOT NULL
        GROUP BY 
            answer_id
    ) b ON a.answer_id = b.answer_id 
    SET 
        a.pipeline_output = b.output_json,
        a.pipeline_output_details = b.full_json_answer_id,
        a.job_status='Processed'
    WHERE 
        a.pipeline_job_id = {pipeline_job_id}
        """
    exec_sql(sql_update)

def update_topics (question_id, pipeline_job_id):
    
    # SQL query to fetch the required data
    sql_update = f"""
        UPDATE ai_pipeline.ai_topic_llm a 
        JOIN (
        SELECT 
            answer_id,
            ans_txt,
            CONCAT(
                '[',
                GROUP_CONCAT(
                    DISTINCT CONCAT(
                        '"',
                        CASE 
                            WHEN group_label_auto IS NULL THEN 'N/A' 
                            ELSE CONCAT( master_insight_agg, ' : ',master_insight, ' : ', insight) 
                        END,
                        '"'
                    ) SEPARATOR ', '
                ),
                ']'
            ) AS output_json,
            MAX(full_json_answer_id) AS full_json_answer_id
        FROM 
            ai.ai_prep_data_llm_final
        WHERE 
            question_id = {question_id}
        GROUP BY 
            answer_id
    ) b ON a.answer_id = b.answer_id 
    SET 
        a.pipeline_output = b.output_json,
        a.pipeline_output_details = b.full_json_answer_id,
        a.job_status = 'Processed'
    WHERE 
        a.pipeline_job_id = {pipeline_job_id}
    """
    print_debug(sql_update)
    exec_sql(sql_update)

def update_group_label (question_id, pipeline_job_id):
    # SQL query to fetch the required data
    sql_update = f"""
    UPDATE ai_pipeline.ai_group_label_llm a 
        JOIN (
            SELECT 
                answer_id,
                ans_txt,
                CONCAT(
                    '[',
                    '"',
                    GROUP_CONCAT(
                        DISTINCT CASE 
                            WHEN master_insight IS NULL THEN 'N/A' 
                            ELSE concat(master_insight_agg, ':' , master_insight) 
                        END SEPARATOR ' :: '
                    ),
                    '"',
                    ']'
                ) AS output_json,
                MAX(full_json_answer_id) AS full_json_answer_id
            FROM 
                ai.ai_prep_data_llm_final
            WHERE 
                question_id = {question_id}
            GROUP BY 
                answer_id
        ) b 
        ON a.answer_id = b.answer_id 
        SET 
            a.pipeline_input  = b.ans_txt,
            a.pipeline_output = b.output_json,
            a.pipeline_output_details = b.full_json_answer_id,
            a.job_status = 'Processed'
        WHERE 
            a.pipeline_job_id = {pipeline_job_id}
        """
    print_debug(sql_update)
    exec_sql(sql_update)

def update_ai_labels_for_raw_text(question_id, pipeline_job_id):
    
    print("generating labels for raw answers from LLM")
   
    sql_update = f"""
        UPDATE ai_pipeline.ai_new_label_llm a 
        JOIN (
            SELECT 
            answer_id,
            JSON_ARRAYAGG(master_insight) AS aggregated_master_insights,
            output_json_details
        FROM (
            SELECT 
                answer_id, 
                ans_txt,
                concat(master_insight_agg, ' : ', master_insight) as master_insight,
                JSON_OBJECT(
                    'split_sentence', split_sentence, 
                    'topic', topic,
                    'group_label', group_label_auto, 
                    'insight', insight, 
                    'master_insight', master_insight,
                    'master_insight_agg', master_insight_agg
                ) as output_json_details
            FROM 
                ai.ai_prep_data_llm_final 
            WHERE 
                question_id = {question_id} AND 
                master_insight IS NOT NULL
        ) AS subquery
        GROUP BY answer_id
        ) b ON a.answer_id=b.answer_id 
        SET 
            a.pipeline_output=b.aggregated_master_insights,
            a.pipeline_output_details=b.output_json_details,
            a.job_status='Processed'
        WHERE a.pipeline_job_id={pipeline_job_id}
    """

    print_debug(sql_update)
    exec_sql(sql_update)

def main(operation, pipeline_job_id, question_id):
    print_debug("in  operation=" + operation +  " pipleine_id=" + str(pipeline_job_id) + " question_id=" + str(question_id))
    if operation == "group_label_insight_llm":
        update_ai_labels(question_id, pipeline_job_id)
    elif operation == "split_text_llm":
        update_split_sentence(question_id, pipeline_job_id)
    elif operation == "get_topic_llm":
        update_topics(question_id, pipeline_job_id)
    elif operation == "get_group_label_llm":
        update_group_label(question_id, pipeline_job_id)
    elif operation == "raw_data_label_llm":
        update_ai_labels_for_raw_text(question_id, pipeline_job_id)
    else:
        print("\n *** WARNING: Wring choice")
    
def unit_test():
    return           
   
if __name__ == "__main__":
    operation=sys.argv[1] # split_sentence, group_label etc
    source_data_type = sys.argv[2] # db or unit_test
    pipeline_job_id=sys.argv[3] # if db , must
    print("operation = " + operation + "source_data_type= " + source_data_type + "pipeline_job_id " + str(pipeline_job_id))
    #question_id=sys.argv[2]
    # record_count=sys.argv[3]
    # if record_count is None:
    #     record_count = 10
    print(len(sys.argv))
    if len(sys.argv) == 4:
        # The first argument in sys.argv[1:] will be your parameter
        main(operation, source_data_type, pipeline_job_id)
    else:
        print("Not enough parameter provided. Running unit test with default settings.")
        



"""
def cluster_ai_labels():
    # work in progress 
    # once all the ai_labels are generated, cluster it based on their 
    # group association to reduce the number of insights into higher level 
    # of categirization. 
    

#step 7, generate a group label by machine 
        # Assuming output_json_arr is your array of JSON objects
        topics_list = [item['topic'] for item in output_json_arr]
        # To see the list of topics
        print(topics_list)
           # Step 8: cluster insights 
        # step 9: provde scoring for each split sentence for generating sample set 


def cluster_topics_old(system_msg, topics_list, parent_topics):
    # Join the list items into a string with the desired format
    json_topics_str = ", ".join([f'"{topic}"' for topic in topics_list])
    # Convert lists to JSON for the prompt
    parent_topics_str = json.dumps(parent_topics)

    # Update the user content with examples
    user_msg = ("You are given an input topic list that is within a code bloc."
            "Your task is to group each topic under a new parent topic. "
            "A prent topic must be <= 3 words"
            "Here is some sample of the parent topics:" + parent_topics_str + "\n"
            "Your task is to generate parent topic and then map the topic to it "
            "The parent topics count should be <= 20% of the topics count.  i.e if there are 40 topics, "
            "you should group them in <= 5 parent topics.  "
            "if question for review is some thing like this, 'What brands you like or dislike', "
            "then try to have parent_topic with brand names if they appear in the feedback"
            "Output the result should ONLY be JSON list in the following format. "
            "[{'parent_topic':<parent_topic1>,'topics':[<topic1>,<topic2>]},'parent_topic':<parent_topic2>:..]}]"
            "The topics list is:\n```" + json_topics_str + "```\n")
    print(user_msg)  
    output_size = len(json_topics_str)*2
    max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
    response = call_openAI(system_msg, user_msg, max_token_estimated)
    pretty_print_json(response)
    # Iterate through the list and print each item
    #for item in response_list:
    #    print(item)
    return response
  
"""
