"""
token costs for 1k tokens: 
 - gpt 3.5 : 0.001653333333
 - gpt 4.0  : 0.00363 
 - gpt 4-turbo: 0.001275

This is the steps in the current LLM process to generate group label, AI labe which Arun is testing with for cold start use case.
Sharing this just as we explore other LLMs, we should also try to see if different steps to improve the outcome even within same model 

use case: cold start: so we have no history, generate trainign data. so the RAG approach is minimal.
So far we have tested with 8-10 studies , and once arun does the gap analysis, of results look good, we will run it for 

step0: generate some context of the client based on metadata [description, vertical, also provisioned terms like brand names, product names] - This will be used for any of the following steps as system context for generating dynamic prompt.  In addition to the system context, each step will have its own context for prompt. 
step 1: 500 raw answer_text -> calculate feedback score [this is based on context of question, like if how to improve, then more negative feedback will have higher score etc]
step 2: get rid of records that is <= threshold
step 3: generate sentiment and intensity [neg, pos, neutral, high, medium , low].  This signal is used for subsequent steps if needed
step 3: generate list of topics for each raw setence with some guardrails to keep # of topics <= 4
step 4: split raw feedback into multiple sentences based on each topic.  Sp if there are 3 topics, it will be 3 setences
step 5: generate actionable insights  for each split sentence with some guardrails and some dynamic context 
step 6: after all the actionable insights generated,  generate group_labels for the entire set of actionable insights and topics associated with it.  
step 7: There is an optional step where user can give a set of group labels, and it will associate each split sentence with a supplied group label.  But this is opitonal. 
step 8: Take all actionable insights from stp6 and cluster them further into lower  number of actionable insights  
step 9:  Constructing training labels and popualte table with 
split sentece 
group_label
AI_label 
other meta data [vertical, sub vertical , question etc] 

step 10: as an additional step, also combine all split setence , and construct final ai_label and associate with original feedback.ans text.  This step is not part of trainign data generation,  This data can be compared with vector db approach of generating final label for the ans_text for the final AI_label.   

Some of the additional steps/ areas of improvement i see 
1. 2-5% of the records it hallucinates, catpruing the reason for it.  
2.  dynamically extracting terms/lexicons using LLM instead of humans supplying it 
3. doing QA progrmitically using counter prompt to assign a quality score instead of totally depending on huma to do QA 

TO BE DONE 
1. integrate client description metadata
2. do a new column for client terms

This script processes and analyzes customer reviews for 'Marico India'. It uses OpenAI's GPT-4 API to interpret the reviews,
extracts insights, sentiments, and categorizes topics. The script also handles database operations for storing and retrieving data.
Functions:
- call_openAI: Communicates with OpenAI API to generate responses.
- get_data_from_db: Fetches reviews from the database based on a question ID.
- save_records_to_db: Inserts processed records into the database.
- cluster_topics: Organizes topics into parent categories.
- cluster_insights: Clusters insights from reviews.
- extract_cluster_from_response: Extracts cluster labels from API responses.
- get_token_size: Calculates token size for API requests.
- merge_parent_topic: Merges insights with parent topics.
- cluster_insights: Cluster the insights into higher level categorization 
- main: Coordinates the review analysis process.
- unit_test: Tests the entire script with a set of predefined reviews.
- more function will come

-- this is run differently than the rest. 
-- schdule following jobs : 
run_all -> 
PREP-DATA (output sentiment, score) -- 
topics TOPICS-LLM: (output: split_sentence, topic)
split  SPLIT-LLM (output: [{split_setence: <split_sentence>, 'group_label':group_label_auto, 'insight': <insight>, 'master_insight':<master_insight>},{},..] )
group_label: GROUPLBL-LLM, (output: )
master_insight: 

** HOW TO RUN THE PROGRAM **
from cli: python3 prep_data_with_llm.py <db or unit_test> <question id> <record limit>
for unit test, rest of the parameters are ignored

"""


import openai
from openai import OpenAI
import datetime
import json
import time
import sys
import pymysql
import os
from collections import Counter
from utils import pretty_print_json, exec_sql, print_debug
#openai.api_key = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
MAXIMUM_THRESHOLD_MASTER_INSIGHT=10 # number of master insight

"""
    Communicates with OpenAI's GPT-4 API to get a response for the given messages.
    Input:
        system_message: String, a message setting the context for the AI.
        user_message: String, the query or task for the AI to respond to.
        max_token_size: Integer, the maximum size of the response in tokens.
    Output:
        A JSON response from the API or a default 'N/A' response upon failure.
"""
def call_openAI(system_message, user_message, max_token_size):
    # Set temperature and max_tokens
    temperature = 0.0  # Adjust the temperature
    model_name = "gpt-4-1106-preview"
    #model_name = "gpt-3.5-turbo"
    #response_format={ "type": "json_object" },           
    max_retries = 3
    if max_token_size is None:
        max_token_size=100
   
    client = OpenAI()

    for attempt in range(max_retries):
        try:
            
            response = client.chat.completions.create(
                model=model_name,
                response_format={ "type": "json_object" },     
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_token_size
            )
            # Extract the response content
            response_content = response.choices[0].message.content
            print_debug("Response Content:" + str(response_content))  # Debugging line

            # Check if response is in Markdown code block and extract JSON
            if response_content.startswith("```json") and response_content.endswith("```"):
               json_content = response_content[7:-3].strip()
            else:
               json_content = response_content
            
            result = json.loads(json_content)
            return result

            
        except json.JSONDecodeError as e:
            print_debug(f"Failed to decode JSON response. Error: {e}. Retrying ({attempt+1}/{max_retries})...")
            print_debug("---------------Error logging starts ----------------")
            print_debug("max token size passed: " + str(max_token_size))
            print_debug("user_msg=\n" + user_message)
            print_debug("\nResponse causing error:\n" + str(response))
            print_debug("---------------Error logging ends ----------------\n")

            # increase the max token to see if that's causing the failure
            max_token_size = int(max_token_size * 1.2)
            if max_token_size > 4090:
                max_token_size=4090

            print_debug("Increasing max_token_size to: " +  str(max_token_size))
            time.sleep(1 + attempt)

        except Exception as e:
            print_debug(f"OpenAI call: An error occurred: {e}")
            time.sleep(1 + attempt)
    
    # If all retries fail, return a default value indicating failure
    print_debug("Maximum retries reached. Returning default 'N/A' response.")
    return {"result": "N/A"}

"""
    Retrieves data from the database for the specified question ID.
    Input:
        question_id: Integer, the ID of the question to which the reviews are associated.
        limit_count: Integer, the number of records to fetch.
    Output:
        A tuple containing customer name, question text, and a list of reviews.
"""
def get_data_from_db(pipeline_job_id, question_id, limit_count):
    # SQL query to fetch the required data
    # prepare data 
    sql_delete = f"DELETE from ai.ai_prep_data_llm WHERE question_id={question_id}"
    exec_sql(sql_delete)

    # bring the data and add some metadata. client_brands is more context.
    sql_insert = f"""
    INSERT into ai.ai_prep_data_llm(
            client_name,client_description,client_brands,
            pipeline_job_id,question_id,question,answer_id,ans_txt)
        SELECT 
            d.name,d.description, null, a.pipeline_job_id,a.question_id, e.question, 
            a.answer_id, a.pipeline_input as ans_txt
        FROM ai_pipeline.ai_prep_data_llm a 
        JOIN triestai.questions  e on e.question_id = a.question_id
        JOIN triestai.factors c on e.factor_id=c.factor_id 
        JOIN triestai.forms b on b.root_factor_id=c.parent_id
        JOIN triestai.clients d on d.client_id=b.client_id
        AND a.pipeline_job_id = {pipeline_job_id}   
        limit {limit_count}
    """
    exec_sql(sql_insert)

    # SQL query to fetch the required data
    sql_fetch = f"""
    SELECT a.client_name,client_description, a.question, a.answer_id, a.ans_txt
    FROM ai.ai_prep_data_llm a
    WHERE question_id={question_id}
    limit {limit_count}
    """
    
    # Execute the SQL query
    status, result = exec_sql(sql_fetch)
    if not status:
        return None, None

    # Initialize the variables
    question_txt = ""
    customer_name = ""
    customer_description = ""
    reviews = []

    # Iterate over the result set
    for row in result:
        if question_txt is not None:
            customer_name = row[0]  # Set the question text
            customer_description = row[1]  # Set the question text
            question_txt = row[2]  # Set the question text
        answer_id = row[3]
        ans_txt = row[4]
        reviews.append([answer_id, ans_txt])

    # Debug information
    if len(reviews) == 0:
        print_debug("** WARNING: No reviews fetched")
        print_debug(sql_fetch)
    else:
        print_debug("set of reviews to be processed:")
        print_debug(reviews)
    print_debug("customer details: " + customer_name + " " + customer_description) 
    return customer_name, customer_description, question_txt, reviews

"""
    Inserts processed records into the database.
    Input:
        question_id: Integer, the ID of the question to which the reviews are associated.
        results: List of dictionaries, each containing processed review data.
    Output:
        None, but performs database insert operations.
"""
def save_records_to_db(results):
    
    for record_list in results:
        for record in record_list:
            # Skip the record if 'ans_txt' is blank
            print_debug("in save, ans_txt = " + record['ans_txt'])
            if 'ans_txt' in record and not record['ans_txt'].strip() or record['ans_txt'].strip() == '':
                print_debug("\n ********* WARNING ans text is missing ***")
                continue

            # Convert the record to a JSON string for the full_json column
            #full_json_str = json.dumps(record).replace("'", "\\'")
            full_json_str = json.dumps(record, ensure_ascii=False)
            full_json_str_for_sql = full_json_str.replace("'", "\\'")

            # Add full_json to the record
            #record['full_json'] = full_json_str  # New line

            # Construct the column names and values for the INSERT statement
            columns = ', '.join(record.keys()) + ', full_json'
            values = ', '.join(["'" + str(value).replace("'", "''") + "'" for value in record.values()])
            values += ", '" + full_json_str_for_sql + "'"

            # Construct the SQL INSERT statement
            sql_insert = f"INSERT INTO ai.ai_prep_data_llm_final ({columns}) VALUES ({values})"

            # Execute the SQL statement
            status, result = exec_sql(sql_insert)
            if not status:
                print_debug(f"Failed to insert record: {record}")
            else:
                print_debug(f"Successfully inserted record with answer_id: {record.get('answer_id', 'N/A')}")

# post processing of individual inserts when done
def save_records_to_db_post_processing(question_id):
    print_debug("update the aggregate columns for answer id level full json")
    sql_fix_json=f"""
        UPDATE ai.ai_prep_data_llm_final 
        SET full_json = REPLACE(full_json, '\n', '\\n')
        WHERE question_id={question_id}
        """
    exec_sql(sql_fix_json)
    
    sql_agg_update = f"""
        UPDATE ai.ai_prep_data_llm_final a join (   
            SELECT 
                b.question_id,
                b.answer_id,
                JSON_OBJECT(
                    'topics', JSON_ARRAYAGG(b.topic),
                    'split_sentences', JSON_ARRAYAGG(b.split_sentence),
                    'group_labels', JSON_ARRAYAGG(b.group_label),
                    'insights', JSON_ARRAYAGG(b.insight),
                    'master_insights', JSON_ARRAYAGG(b.master_insight)
                ) AS full_json
            FROM 
                ai.ai_prep_data_llm_final b
            where b.question_id={question_id}
            GROUP BY 
                b.answer_id 
        ) c on a.answer_id=c.answer_id
        SET a.full_json_answer_id=c.full_json;
    """
    exec_sql(sql_agg_update)

def save_output_data_to_db(question_id, pipeline_id):
    # this code updates data in the orignal source table that CRM populated
    print_debug("\n ---- Step 13: Updating final input table for pipeline_out columns")
    
    sql_update_data=f"""
        UPDATE ai_pipeline.ai_prep_data_llm a 
        JOIN (
            SELECT 
                answer_id, 
                JSON_ARRAY(
                    CONCAT(
                        MAX(sentiment), 
                        ' :: ',
                        ROUND(MIN(answer_score), 2),
                        ' :: ',
                        GROUP_CONCAT(DISTINCT topic SEPARATOR ', ')
                )
            ) AS json_column,
                MAX(full_json_answer_id) AS modified_full_json
            FROM 
                ai.ai_prep_data_llm_final 
            WHERE question_id = {question_id}
            AND JSON_VALID(full_json_answer_id)
            GROUP BY answer_id
        ) b 
        ON a.answer_id = b.answer_id
        SET a.pipeline_output = b.json_column, 
            a.pipeline_output_details = b.modified_full_json,
            a.job_status='Processed'
        WHERE a.question_id ={question_id}
        AND pipeline_job_id={pipeline_id}
    """

    exec_sql(sql_update_data)

def update_progress_level_in_db(pipeline_job_id, progress_level):
    # update progress level in db
    sql_update=f"""UPDATE ai.ai_pipeline_jobs 
                SET job_progress_level = {progress_level}
                WHERE pipeline_job_id = {pipeline_job_id}
            """
    exec_sql(sql_update)

"""
Processes a list of topics and organizes them into parent categories based on predefined 
rules. It supports batching for large datasets and deduplication to ensure unique parent topics.
"""
def cluster_topics(system_msg, topics_list, parent_topics):

    print_debug("Number of topics before de-duplication: " + str(len(topics_list)))
    # De-duplicate the topics_list
    topics_list = list(set(topics_list))
    print_debug("Number of topics after de-duplication:" + str(len(topics_list)))

    def send_request(topics):
        json_topics_str = ", ".join([f'"{topic}"' for topic in topics])
        parent_topics_str = json.dumps(parent_topics)
        user_msg = ("You are given an input topic list that is within a code bloc. "
                    "Your task is to group each topic under a new parent topic. "
                    "A parent topic must be <= 3 words "
                    "Here is some sample of the parent topics:" + parent_topics_str + "\n"
                    "Your task is to generate parent topic and then map the topic to it. "
                    "The parent topics count should be <= 20% of the topics count.  "
                    "if question for review is some thing like this, 'What brands you like or dislike', "
                    "then try to have parent_topic with brand names if they appear in the feedback. "
                    "Output the result should ONLY be JSON list in the following format. "
                    "[{'parent_topic':<parent_topic1>,'topics':[<topic1>,<topic2>]},'parent_topic':<parent_topic2>:..]}]"
                    "The topics list is:\n```" + json_topics_str + "```")

        # Estimating token size and capping if necessary
        total_token_size = len(user_msg) + len(system_msg)
        #max_token_estimated = min(total_token_size, 4096)
        output_size = len(json_topics_str)*3
        max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
        response = call_openAI(system_msg, user_msg, max_token_estimated)
        return response

    # dividing batches if needed, because output_token is exceeding the threshold of 
    token_conversion_ratio = 0.25  # Approximate conversion ratio of characters to tokens
    # Calculate total size in tokens and average token size per topic
    total_size_in_tokens = len(json.dumps(topics_list)) * token_conversion_ratio
    average_token_size_per_topic = total_size_in_tokens / len(topics_list)
    buffer_factor = 0.67  # 33% buffer
    batch_size = int((4096 * buffer_factor) / average_token_size_per_topic)

    print_debug("total list of topic = " + str(len(topics_list)) + " total token size = " + str(total_size_in_tokens))
    # Check if batching is necessary
    if total_size_in_tokens <= 3500:  # keeping some buffer and not checking to threshold of 4096
        print_debug("total output size is within allowed limit, not dividing in batches")
        return send_request(topics_list)
    else:
        # Split topics into dynamic batches
        print_debug("total output size exceeded allowed limit, dividing in batches")
        print_debug("batch size = " + str(batch_size))
        topic_batches = [topics_list[i:i + batch_size] for i in range(0, len(topics_list), batch_size)]
        combined_results = []

        for batch in topic_batches:
            batch_results = send_request(batch)
            if batch_results == 'N/A' or not isinstance(batch_results, list):
                print_debug("Unhappy path encountered, skipping this batch.")
                combined_results.append({"parent_topic": "N/A", "topics": []})  # Handle unhappy path
            else:
                combined_results.extend(batch_results)

        # Deduplicate parent topics
        final_results = {}
        for item in combined_results:
            if item['parent_topic'] == 'N/A':
                continue  # Skip processing this item
            parent_topic = item['parent_topic']
            topics = final_results.get(parent_topic, [])
            topics.extend(item['topics'])
            final_results[parent_topic] = list(set(topics))  # Deduplicate topics under each parent

        # Convert to desired output format
        final_output = [{'parent_topic': k, 'topics': sorted(v)} for k, v in final_results.items()]
        return final_output


"""
Clusters insights from reviews based on their content and relevance. It uses OpenAI's API for clustering 
and ranks insights based on feedback scores and sentiment.
"""
def cluster_insights(insights_data, system_msg):
    print_debug("in cluster ingihts")
    pretty_print_json(insights_data)
    
    clustered_data = []

    # Group insights by parent topic
    groups = {}
    for sublist in insights_data:  # Iterate over each sublist
        for record in sublist:  # Iterate over each record in the sublist
            # group = record["group_label_auto"]
            # Use 'group_label_auto' if it exists, otherwise default to 'Unknown'
            group = record.get("group_label_auto", "Unknown")
            if group not in groups:
                groups[group] = []
            groups[group].append(record)

    print_debug("in cluster insights")
    clustered_data = []

    # Process each group
    for group, insights in groups.items():
        # Sort insights within the group by feedback score and sentiment
        insights.sort(key=lambda x: (x["answer_score"], x["sentiment"]), reverse=True)

        # Prepare the prompt for OpenAI API
        user_msg = ("You are given a list of insights, and your job is to reduce the count of them.  "
                    "Please cluster the multiple insights that are similar into meaningful one insight. "
                    "Example of grouping: 'Fix delivery delay' , 'Delay in delivery' These two can be grouped into an insight 'Fix delivery delay' "
                    "The final insight have to be an actionable insight and not a high level grouping. "
                    "The total number of final insights should be aorund 3-6 maximum. "
                    "If you have only one insight for the group, just make the master_insight same as insight. "
                    "In case the insights cannot be grouped because they are widely different, order the insights "
                    "in terms of importance with first insight in a group is the most important "
                    "Provide the output in JSON format with a key: insights_clustered that has sub_keys: master_insight, child_insights_list. "
                    "master_insight: the clustered insight and child_insights_list contains list of insights that is related to the master_insight. "
                    "Here are the insights:\n")

        for insight in insights:
            user_msg += f"- {insight['insight']}\n"
        user_msg += "\nFormat the output as a JSON array where each element has the insight and its cluster label."
        print_debug(user_msg)

        output_size = len(user_msg)*1.25
        total_token_size = get_token_size(len(user_msg), len(system_msg), output_size)
        max_token_estimated = min(total_token_size, 4096)
        result = call_openAI(system_msg, user_msg, max_token_estimated)
        
        response_text = result.get('insights_clustered','N/A')
        # if response_text != 'N/A':
        #     if isinstance(response_text, str):
        #         # Parse the JSON string to a Python object
        #         cluster_info = json.loads(response_text)
        #     else:
        #         cluster_info = response_text  # Assuming it's already a Python object
        # else:
        #     # Handle the case where response_text is 'N/A'
        #     print_debug("No valid response from OpenAI, skipping the clustering process.")
        #     continue  # Returning the original data as no clustering can be performed

        if response_text:
            try:
                if isinstance(response_text, str):
                    cluster_info = json.loads(response_text)
                else:
                    cluster_info = response_text  # Assuming it's already a Python object
            except json.JSONDecodeError:
                print_debug("Error decoding JSON response, setting default clustering.")
                cluster_info = [{"master_insight": "N/A", "child_insights_list": [insight["insight"] for insight in insights]}]
        else:
            print_debug("No valid response from OpenAI, setting default clustering.")
            cluster_info = [{"master_insight": "N/A", "child_insights_list": [insight["insight"] for insight in insights]}]


        pretty_print_json(cluster_info)
        #cluster_info = json.loads(response_text)
        clustered_data.append(cluster_info)

    print_debug("\n ----- clustering insights done for all group labels")
    pretty_print_json(clustered_data)
        
    print_debug("\n ----- merging master insight to the main array")
    # now merge master_ionsight to the main insight 
    # Create a mapping from child insight to master insight
    insight_mapping = {}
    for cluster_group in clustered_data:  # Iterate over each cluster group
        for item in cluster_group:  # Iterate over each item in the cluster group
            if isinstance(item, dict):  # Check if item is a dictionary
                master_insight = item.get("master_insight")
                if master_insight:  # Ensure master_insight exists
                    #print_debug("master insight:\n" +  str(master_insight))
                    for child_insight in item.get("child_insights_list", []):
                        insight_mapping[child_insight] = master_insight
            else:
                print_debug("\n ********* WARNING: fail to get master_insight from item var as it is not a dict")
    # Update array1 with the master_insight
    for sublist in insights_data:
        for record in sublist:
            child_insight = record.get("insight")
            if child_insight in insight_mapping:
                record["master_insight"] = insight_mapping[child_insight]

    pretty_print_json(insights_data)
    return insights_data

"""
Extracts cluster information from OpenAI's API response, 
matching each insight with its corresponding cluster label.
"""
def extract_cluster_from_response(cluster_info, insight):
    """
    Extracts the cluster information from the structured JSON response.
    Each element in cluster_info is expected to have the insight and its cluster label.
    """
    for cluster in cluster_info:
        if cluster.get('insight') == insight:
            return cluster.get('cluster', 'Unknown')
    return 'Unknown'

"""
Calculates the appropriate token size for OpenAI API requests based on input and output message lengths. 
It ensures requests adhere to token limits and adjusts for potential buffer.
"""
def get_token_size(user_message_size, system_message_size,output_size):
    buffer=1.4
    token_conversion=0.25
    max_token_size_input=round((user_message_size+system_message_size) * buffer * token_conversion)
    max_token_size_output=round((output_size) * buffer * token_conversion)
    #print_debug("max token:" + str(max_token_size_output))
    if max_token_size_output > 4096:
        return 4090
    elif max_token_size_output <= 150:
        return 150
    else:
        return max_token_size_output

def merge_parent_topic(array1, array2):
    """
    Merges insights from individual reviews with their associated parent topics. 
    This function enriches the dataset by adding context to each review insight.
    """
    #array2 has whole review
    array3 = array1["parent_topics"]
    # Create a mapping from topic to parent topic
    topic_to_parent = {}
    
    if isinstance(array1, str):
        try:
            array1 = json.loads(array1)
        except json.JSONDecodeError as e:
            print_debug(f"Error decoding JSON: {e}")
            return None

    for group in array3:
        
        if isinstance(group, dict) and "parent_topic" in group:
            parent_topic = group["parent_topic"]
            for topic in group["topics"]:
                topic_to_parent[topic] = parent_topic
        else:
            print_debug(f"Invalid group structure: {group}")
            return None

    # Loop through array2 and add parent topic to each record
    for sub_array in array2:
        for record in sub_array:
            topic = record.get("topic")
            if topic in topic_to_parent:
                record["group_label_auto"] = topic_to_parent[topic]

    # array2 now has the parent_topic added to each record
    return array2

def process_review(review_arr, question_id, system_msg, question):
    """
    Orchestrates the overall process of analyzing individual reviews. 
    It sequentially performs steps like scoring feedback, extracting sentiment, 
    and clustering insights, assembling a comprehensive analysis of each review.
    """
    review_id = review_arr[0]
    review_text = review_arr[1]
   
    # 0
    print_debug("\n -------- 1st step - feedback score ------------")
    # User message with instructions for the model
    user_msg = ("Please analyze the following feedback text enclosed within the code block and "
                "generate a score. Output result in a structured JSON with "
                "1 key, feedback_score. A feedback score gives an idea how important "
                "is the feedback and the score should be more if actionable insight, has "
                "adequate details. In general, negative feedback should have a higher score than "
                "positive feedback. The score should be between 0.1 to 0.99. "
                "Here is the text: \n\n```" + review_text + "```")


    output_size=200
    max_token = get_token_size(len(user_msg), len(system_msg), output_size)
    
    result = call_openAI(system_msg, user_msg, max_token)
    feedback_score = result.get('feedback_score','N/A')
    if feedback_score == "N/A":
        print_debug("\n **** WARNING , issues with feedback,  skipping it")
        return None
    
    print_debug("feedback score = " + str(feedback_score))

    # if feedback is of poor quality, skip
    if feedback_score != 'N/A' and feedback_score <= 0.2:
        print_debug("\n ********* WARNING: Poor feedback score, < 0.2, skipping  ********")
        return None
    
    #1
    print_debug("\n -------- 2nd step - extract sentiment ------------")

    output_size=200
    max_token = get_token_size(len(user_msg), len(system_msg), output_size)
    #max_token=400
    
    user_msg = ("Extract the sentiment from the following review enclosed within a code block: "
                "Output result in a strucred JSON with 1 key: sentiment. "
                "Each review only have 1 sentiment"
                "here is the review \n\n```" + review_text + "```")
    
    result = call_openAI(system_msg, user_msg,max_token)
    print_debug("Parsed Response:" + str(result))
    
    #sentiment = result['sentiment']
    #print_debug("sentiment = " + sentiment)
    # Debugging: Print the structure of the result
    try:
        # Assuming the structure is as expected. Adjust the key access as per actual structure.
        sentiment = result.get('sentiment', "N/A")
        print_debug("Sentiment = " + sentiment)
    except KeyError:
        print_debug("Key 'sentiment' not found in the response:", result)
    
    
    #2
    print_debug("\n -------- 3rd step topic list ------------")
    
    output_size=200
    max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
    max_token=500
    
    user_msg = ("Extract the list of topics from the following review enclosed within a code block. " 
                "Output result in a structured JSON with key "
                "list_of_topics and a sub key topic that has the topic description. "
                "Order the topics in terms of their importance, "
                "so first topic should have the highest importance. "
                "A topic MUST be <= 4 words"
                "A feedback must NOT have more than 3 topics. "
                "If you need to have a 4 or more topics, then add an 'Others' topic as a catch-all as a fourth topic"
                "If a feedback is short, try to keep number of topics also less."
                "If a feedback is short, like less than 20 words, keep the number of topics <= 2"
                "Output the results in a JSON format "
                "here is the review \n\n```" + review_text + "```")
    
    result = call_openAI(system_msg, user_msg, max_token_estimated)
    list_of_topics = result.get('list_of_topics','N/A')
    print_debug("\n *** list of topic: " + str(list_of_topics) + " count of topics : " + str(len(list_of_topics)) + " *********\n")
    print_debug(list_of_topics)

    if len(list_of_topics) == 0 or list_of_topics == 'N/A' or not list_of_topics:
        print_debug("\n ********* WARNING: No topics found, skipping  ********")
        return None

    #4
    print_debug("\n -------- 4th step split sentence list ------------")
    user_msg = ("Please analyze the following review text enclosed within the code block "
                    "and split the text into multiple sets based on each topic. "
                    "Each set has a part of the feedback text, must be corrensponding to only one topic. "
                    "Remove any delimeter word like 'and' or 'but' if needed at beginning or end after the split from the individual sets. "
                    "Order the sets in terms of their position in the feedback, "
                    "so first set should be from the begining of the feedback. "
                    "Output the results in a nested JSON format with a key: split_sentence_list and sub keys will be topic and split_text "
                    "The list of topics in the text is: \n " + str(list_of_topics) + "\n\n"
                    "and the review you need to process:\n ```" + review_text + "```") 
           
    
    output_size=len(user_msg)*1.5
    max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
    #max_token=900
    
    result = call_openAI(system_msg, user_msg, max_token_estimated)
    split_sentence_list = result.get('split_sentence_list','N/A')
    print_debug(split_sentence_list)
    if split_sentence_list == 'N/A':
        print_debug("No valid response from OpenAI for split sentence list, proceeding with default or next steps.")
        return None

    output_json_arr =[]
    
    for item in split_sentence_list:
        split_text = item["split_text"]
        topic = item["topic"]

        #5 get actionable insight for the split text :
        print_debug("\n 5th step --------- getting actionable insght for split text ---------") 
        print_debug("split text = " + split_text)
        # Check if split_text is blank or less than 10 characters
        if not split_text or len(split_text) < 10:
            print_debug("split text is blank or < 10 chars, skipping")
            continue  # Skip to the next iteration
        
        # cold start usecase, with hot-start, we should generate some past data thru RAG to enrich the context.
        user_msg = ("Please analyze the following text enclosed within the code block and extract an actionable insight. "
            "The insight should be NO More than 7 words. "
            "Present your analysis in a structured JSON format, with the key 'insight' "
            "holding the extracted information. Here is the text:\n\n```" + split_text + "```")

        #print_debug("user msg for #5 : \n" + user_msg + "\n")
        output_size=100
        max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
        
        result = call_openAI(system_msg, user_msg, max_token_estimated)
        insight = result['insight']
        print_debug(insight)

        print_debug("\n 6th step --------- getting group label for topic: " + topic + "  ---------") 
        group_label_list = ["Product issue", "Delivery Issue", "Customer Service",\
                            "Pricing issue", "Positioning/Branding","Supply chain issue", \
                                "Transportation", "Branding/Marketing", "Store Experieince", "Competition", "Others"]
        #group_label_list = ["Delivery Issue","Pricing issue"]
        user_msg = ("Asscoiate the topic enclosed within a code block "
                    "with a pre-defined category from a list of categories. "
                "The list of categories are\n" + str(group_label_list) + "\n"
                "Output the result in a structured JSON with key: 'group_label'"
                "The topic  is ```" + topic + "```\n")
        
        output_size=200
        max_token_estimated = get_token_size(len(user_msg), len(system_msg), output_size)
        result = call_openAI(system_msg, user_msg, max_token_estimated)
        group_label = result.get('group_label','N/A')
        
        if group_label is None:
            print_debug("\n ********* WARNING: Group label not found, skipping  ********")
            return None 
        
        print_debug("group label for topic: " + topic + " , group label: " + str(group_label))

        output_json = {
        "question_id":question_id,
        "question": question,
        "answer_id":review_id,
        "ans_txt":review_text,    
        "answer_score": feedback_score,
        "sentiment": sentiment,
        "group_label": group_label,
        "topic": topic,
        "split_sentence": split_text,
        "insight": insight
        }
        print_debug("arr for split sentence")
        #pretty_print_json(output_json)

        output_json_arr.append(output_json)
    
    print_debug("\n\n ---- final array for review = ---- " + str(review_id))
    pretty_print_json(output_json_arr)

    return output_json_arr

def call_openAI_for_aggregated_insights(system_msg, master_insights):
    """
    Call OpenAI API to generate aggregated master insights.
    Parameters:
    - system_msg: System message to provide context for the OpenAI model.
    - master_insights: A list of master insights to be aggregated.
    Returns:
    - Aggregated insights in JSON format or None if the API call fails.
    """

    # Enhanced user message with detailed output instructions
    user_msg = (
        "Please analyze the following list of master insights and aggregate them into higher-level insight. "
        "Each insight should be grouped based on its thematic similarity to others. "
        "Output the result in a structured JSON format with a key 'aggregated_insights'. "
        "Here is the list of master insights: \n\n" +
        json.dumps(master_insights, indent=2)
    )
    print_debug("call_openAI_for_aggregated_insights: user_msg \n" + user_msg)
    output_size = len(user_msg) # since output will have the entire nested array... 

    max_token = get_token_size(len(user_msg), len(system_msg), output_size)
    
    result = call_openAI(system_msg, user_msg, max_token)
    aggregated_insights = result.get('aggregated_insights', 'N/A')

    if aggregated_insights == "N/A":
        print_debug("\n **** WARNING, issues with aggregation, skipping it")
        return None

    #print_debug("call_openAI_for_aggregated_insights: Aggregated Insights: \n" + str(aggregated_insights))
    return aggregated_insights

def process_nested_array_and_aggregate_insights(nested_array, system_msg):
    # Extract all master_insights
    master_insights = set()
    for array in nested_array:
        for record in array:
            if isinstance(record, dict) and "master_insight" in record:
                master_insights.add(record["master_insight"])
    print_debug("master insights array")
    print_debug(master_insights)

    if len(master_insights) > MAXIMUM_THRESHOLD_MASTER_INSIGHT:
        # Step 2: Use OpenAI to generate aggregated insights
        aggregated_insights = call_openAI_for_aggregated_insights(system_msg, list(master_insights))
        print_debug("process_nested_array_and_aggregate_insights: aggregated_insights \n" )
        pretty_print_json(aggregated_insights)
        if not aggregated_insights:
            print_debug("Failed to obtain aggregated insights.")
            return nested_array

        master_to_agg_insight = {}
        for agg_insight, master_list in aggregated_insights.items():
            for master in master_list:
                master_to_agg_insight[master] = agg_insight

        # Map and update records with master_insight_agg
        for array in nested_array:
            for record in array:
                master_insight = record.get('master_insight')
                if master_insight in master_to_agg_insight:
                    record['master_insight_agg'] = master_to_agg_insight[master_insight]
                else:
                    if master_insight:
                        print_debug("master insight: " + master_insight + " not found in master insight agg array")
                    record['master_insight_agg'] = 'Unknown'
    else:
        print_debug("Skipping master insight agg; count of master insight < threshold")
        for array in nested_array:
            for record in array:
                record['master_insight_agg'] = record.get('master_insight', 'Unknown')

    return nested_array

def construct_system_message(customer_name, question):
    # Define prompts to get company details and brands
    prompt_details = f"Provide a brief summary of the company named {customer_name}. Please output your response in 2 lines json format with a key 'customer_details'"
    prompt_brands = f"List the main brands associated with the company named {customer_name}. Please output your response in 2 lines json format with a key 'customer_brands'"

    # Call OpenAI to get responses
    output_size = 100  # Adjust as needed
    max_token_details = get_token_size(len(prompt_details), 100, output_size)
    max_token_brands = get_token_size(len(prompt_brands), 100, output_size)

    system_msg_tmp = "you are an AI assistant"
    response_details = call_openAI(system_msg_tmp, prompt_details, max_token_details)
    response_brands = call_openAI(system_msg_tmp, prompt_brands, max_token_brands)

    # Directly access keys from response dict
    customer_details = response_details.get('customer_details', 'Details not available')
    customer_brands = response_brands.get('customer_brands', 'Brands not available')

    # Construct the system message
    system_msg = f"""You are a helpful assistant that analyzes reviews. The reviews are for customer: {customer_name}. 
            
            Description of customer: 
            {customer_details}. 
            
            These are customer brands and products: 
            {', '.join(customer_brands)}.

            The text of the question may have some information about the company vertical 
            or specialization that will provide the context of the reviews. 
            The reviews are mostly around sales and customer issues. 
            The review needs to be processed through a series of steps 
            to get actionable insights at the end. 
            The question asked for which reviews need to be processed: 
            {question}
            """

    return system_msg

"""
Serves as a testing function to validate the entire process using a predefined set of reviews. 
It simulates the script's execution and provides an opportunity to debug and assess functionality.
record count is mostly used for unit testing from cli
set a defaul rec count=1000
"""
def main(operation, pipeline_job_id, question_id, record_count=500):
    start_msg = f"{datetime.datetime.now().strftime('%H:%M:%S')} - Started \n"
    print_debug(start_msg)
    print_debug(operation + ", pipeline_job_id:" + str(pipeline_job_id) + " question id:" + str(question_id) + " rec count:" + str(record_count))
        
    if int(record_count) > 1000:
        print_debug("\n\n ************ WARNING: MORE THAN 1K RECORD IS THERE, WILL CAP IT TO 1K")
    customer_brands = ""
    customer_details = ""

    if operation == "unit_test":
       customer_name, question_id, question, reviews = unit_test()

    else:
        customer_name, customer_details, question, reviews = get_data_from_db(pipeline_job_id, question_id,record_count)
        print_debug("number of reviews = " + str(len(reviews)))
        
        if len(reviews) == 0:
            print_debug("\n *** WARNING ** No reviews")
            return
        pretty_print_json(reviews)

        # this infor needs to come from customer metadata to build a rich context for prompt
        customer_details = "Info not available"
        customer_brands  = "Info not available"
        
    print_debug("\n ---- starting , total count of reviews = " + str(len(reviews)) + "----- ")
    
    system_msg = construct_system_message(customer_name, question)
    print_debug("system msg = \n" + system_msg)

    result_final = []
    result_bad_data = []
    counter = 0
    # Initialize the variable outside the loop
    last_recorded_percentage = -1
    
    for customer_review in reviews:
        counter += 1
        progress_level = counter/len(reviews)
        print("review=\n" + str(customer_review[1]))

        print_debug("\n--------- Review, counter = " + str(counter) + " of total count " + str(len(reviews))  + " Progress level " + str(progress_level) + " ---------------\n\n")
        # keep some buffer for further processing after all reviews get over
        if progress_level > 0.9:
            progress_level = 0.9

        # Calculate the percentage of progress as an integer
        progress_percentage = int((counter / len(reviews)) * 100)
        # Check if the current percentage is a multiple of 10 and is greater than the last recorded percentage
        if progress_percentage % 10 == 0 and progress_percentage > last_recorded_percentage:
            print_debug("\n********* Progress level " + f"{progress_percentage}% completed." + " **************")
            update_progress_level_in_db(pipeline_job_id, progress_percentage / 100)
            last_recorded_percentage = progress_percentage  # Update the last recorded percentage

        print_debug(customer_review)
        if len(customer_review[1]) <= 10: # this needs to be looked at, as one word feedback sometimes can be meaniful
            print_debug("skipping review as very less size")
            result_bad_data.append(customer_review)
        else:
            print_debug("-----------------\n")
            result_arr = process_review(customer_review, question_id, system_msg, question)
            if result_arr is not None:
                result_final.append(result_arr)
            else:
                result_bad_data.append(customer_review)
    
    pretty_print_json(result_final)

    print_debug("\n --------------- step 8 - clustering -----")
    # clustering can only be done after all the insights generated as opposed inside 
    # the loop of earlier step.
    topics_list = []
    for sub_array in result_final:
        for record in sub_array:
            if 'topic' in record:
                topics_list.append(record['topic'])
    #print_debug(topics_list)
    
    # Example specific topics and parent topics
    parent_topics = ["Price", "Customer Service", "Delivery Issue", "Product Issue", "Supply chain", "Customer Demand", "Branding/Marketing","Store Experience", "Promotions","Others"]
    clustered_topics = cluster_topics(system_msg,topics_list,parent_topics)
    # now add parent_topic to the list 
    results_with_parent_topic = merge_parent_topic(clustered_topics, result_final)
    pretty_print_json(results_with_parent_topic)

    # now cluster the insight
    print_debug("\n -------------- step 10: cluster insights ")
    insights_collapsed = cluster_insights(results_with_parent_topic, system_msg)
    print_debug(insights_collapsed)

    print_debug("\n -------------- step 10.5: Agg master insights further")
    master_insights_collapsed = process_nested_array_and_aggregate_insights(insights_collapsed, system_msg)
    print_debug("after agg: master_insights_collapsed\n")
    pretty_print_json(master_insights_collapsed)

    print_debug(master_insights_collapsed)
    insights_collapsed=master_insights_collapsed
    
    print_debug("\n -------------- step 10.6: invert the array for better viewing")
    insights_collapsed_inverted = restructure_nested_array(insights_collapsed)
    
    if operation != "unit_test":
        print_debug("\n -- step 11, savins records in DB --- ")
        delete_sql = f"delete from ai.ai_prep_data_llm_final where question_id={question_id}"
        exec_sql(delete_sql)
        save_records_to_db(insights_collapsed)
        
    print_debug("\n\n *********** All bad data ***********")
    print_debug("before processing: \n")
    pretty_print_json(result_bad_data)

    # Transform each tuple in result_bad_data to a list of dictionaries
    formatted_bad_data = []
    for record in result_bad_data:
        answer_id, ans_txt = record  # Unpacking tuple
        formatted_record = [{
            "answer_id": answer_id,
            "ans_txt": ans_txt,
            "question_id": question_id,
            "question": question,
            "is_bad_data": 1  # Assuming you want to set this flag for bad data
        }]
        formatted_bad_data.append(formatted_record)

    print_debug("after processing: \n")
    pretty_print_json(formatted_bad_data)
    
    if operation != "unit_test":

        # Now pass this formatted list of lists to save_records_to_db
        save_records_to_db(formatted_bad_data)
        print_debug("\n Now updating the ai_pipeline source table")
        save_records_to_db_post_processing(question_id)
        save_output_data_to_db(question_id, pipeline_job_id)

    print_debug("final array")
    for sub_array in insights_collapsed:
        pretty_print_json(sub_array)
    
    print_debug("\n------------------STATS ------------\n")
    summary_stats = summarize_nested_array(insights_collapsed,formatted_bad_data)
    print_debug(summary_stats)
    
    start_msg = f"{datetime.datetime.now().strftime('%H:%M:%S')} - Ended \n"
    return insights_collapsed

# gather summary stats
def summarize_nested_array(nested_array, bad_data_array):
    unique_answer_ids = set()
    split_sentences_count = 0
    unique_group_label_auto = set()
    unique_master_insight = set()
    unique_insight = set()
    total_feedback_score = 0
    master_insight_details_with_split_sentences = {}  # New dictionary
    master_insight_agg_mapping_master_insight = {} 

    # Track the number of answer_ids and split_sentences for each master_insight
    master_insight_stats = {}

    # Iterate through nested array to collect data
    for array in nested_array:
        for record in array:
            answer_id = record["answer_id"]
            master_insight = record.get("master_insight", "Not Available")
            master_insight_agg = record.get("master_insight_agg", "Not Available")

            unique_answer_ids.add(answer_id)
            split_sentences_count += 1
            unique_group_label_auto = record.get("group_label_auto", "Not Available")
            unique_master_insight.add(master_insight)
            insight = record.get("insight", "Not Available")
            split_sentence = record.get("split_sentence", "")  # Assuming there is a key 'split_sentence'
            unique_insight.add(insight)
            total_feedback_score += record.get("answer_score", 0)

            # Update stats for master_insight
            if master_insight not in master_insight_stats:
                master_insight_stats[master_insight] = {"answer_ids": set(), "split_count": 0}
                master_insight_details_with_split_sentences[master_insight] = []  # Initialize list for each master_insight
            master_insight_stats[master_insight]["answer_ids"].add(answer_id)
            master_insight_stats[master_insight]["split_count"] += 1
            master_insight_details_with_split_sentences[master_insight].append(split_sentence)

            if master_insight_agg not in master_insight_agg_mapping_master_insight:
                master_insight_agg_mapping_master_insight[master_insight_agg] = set()
            master_insight_agg_mapping_master_insight[master_insight_agg].add(master_insight)

    # Convert set to list for JSON serialization
    for agg, insights in master_insight_agg_mapping_master_insight.items():
        master_insight_agg_mapping_master_insight[agg] = list(insights)

    # Calculating summary stats
    bad_records_count = len(bad_data_array)
    unique_answer_ids_count = bad_records_count + len(unique_answer_ids)
    avg_feedback_score = total_feedback_score / split_sentences_count if split_sentences_count > 0 else 0
    avg_split_sentences_per_answer = split_sentences_count /  len(unique_answer_ids) if unique_answer_ids else 0
    bad_records_percentage = (bad_records_count / unique_answer_ids_count * 100) if unique_answer_ids else 0
    # Formatting master_insight stats
    master_insight_summary = {mi: (len(stats["answer_ids"]), stats["split_count"]) for mi, stats in master_insight_stats.items()}

    if len(unique_master_insight) > 0:
        count_of_split_sentence_per_master_insight = round(split_sentences_count / len(unique_master_insight), 2)
    else:
        count_of_split_sentence_per_master_insight = 0

    summary = {
        "total_number_of_unique_answer_ids": unique_answer_ids_count,
        "total_number_of_split_sentences": split_sentences_count,
        "average_number_of_split_sentences_per_answer": avg_split_sentences_per_answer,
        "total_number_of_unique_group_label_auto": len(unique_group_label_auto),
        "total_number_of_unique_master_insight": len(unique_master_insight),
        "total_number_of_unique_master_insight_agg": len(master_insight_agg_mapping_master_insight),
        "total_number_of_unique_insight": len(unique_insight),
        "average_feedback_score": avg_feedback_score,
        "count_of_split_senence_per_master_insight": count_of_split_sentence_per_master_insight,
        "percentage_group_label_auto_per_answer_id": f"{len(unique_group_label_auto) / unique_answer_ids_count * 100:.2f}%",
        "number_of_bad_records": bad_records_count,
        "percentage_of_bad_records": f"{bad_records_percentage:.2f}%",
        "list_of_unique_master_insights": list(unique_master_insight),
        "master_insight_details_with_ans_id_and_split_counts": master_insight_summary,
        "master_insight_details_with_split_sentences": master_insight_details_with_split_sentences,
        "master_insight_agg_mapping_master_insight" : master_insight_agg_mapping_master_insight

    }

    return json.dumps(summary, indent=2)

def restructure_nested_array(nested_array):
    new_structure = {}

    for array in nested_array:
        for record in array:
            master_insight_agg = record.get("master_insight_agg", "Not Available")
            master_insight = record.get("master_insight", "Not Available")
            split_sentence = record.get("split_sentence", "Not Available")
            answer_id = record.get("answer_id", "Not Available")
            ans_txt = record.get("ans_txt", "Not Available")

            if master_insight_agg not in new_structure:
                new_structure[master_insight_agg] = {}

            if master_insight not in new_structure[master_insight_agg]:
                new_structure[master_insight_agg][master_insight] = {}

            new_structure[master_insight_agg][master_insight][split_sentence] = {
                "answer_id": answer_id,
                "ans_txt": ans_txt
            }

    return new_structure



def unit_test():
    question_id=1
    review_text = "The product quality is good, but the price is expensive. Also, I faced delivery issues."
    question_text="Please provide all your IDEAS (BIG or SMALL) which will help Reduce Cost or \
        Increase Productivity across functions. Feel free to put in as many IDEAS as you wish..."
    
    review_text="1. My idea is that We should focus on Saffola edible oil packaging . We need to improve the quality \
        of pouches.Its very poor compared to other players in market. This improvisation will help us \
        improve our market share. 2. In Oats Flavour packs are only .43gms. The market is demanding \
        for bigger size packets. I do understand the issue of maintaining the quality in bigger packs. \
        But a innovation in this would help us gain the early bird advantage. 3.I have an IDEA , \
        we can come up with CUP Oats(masala) like the cup Noodles. This would help us enter Eatery channels.  \
        4.We can bring Bulk Breaking for upcountry Midas DDs, This would help us bill slow moving \
        SKUs to tier 2 markets. This would create interest in upcountry DDs to focus \
        on range selling of slow moving SKUs.  5. We should again seriously implement \
        or focus on VMI generated orders for upcountry DDs. This would bring back \
        confidence in our Systems for our DDs.  6. We should focus on educating \
        the retailers, DSRs and end consumers about the Blend, advantages \
        of kardi and merits of using Saffola. Which I think offlate we are not doing .  \
        Thank You  Smithun Chennai Metro TSE Metro 9176686642."
    
    customer_brands = "Some of Lupin Pharmaceuticals' prominent product/brand names include Solosec, Alinia, Suprax, and Methergine, among others."
    customer_name="Lupin Pharmacuticals"
    customer_details="Marico produces hair oil, edible oil etc. and sells its goods through lot of brick-n-mortar store"   
    customer_details="Lupin Pharmaceuticals, India, is a leading pharmaceutical company that specializes in the research, development, and manufacturing of a wide range of high-quality generic and branded pharmaceuticals, as well as biotechnology products. With a global presence and a commitment to providing affordable and innovative healthcare solutions, Lupin is dedicated to improving the health and well-being of people worldwide."
    question = "Please share your expereience in the grocery store"
    question = "Lupin-Pharma-Maxter - What are the TOP TWO reasons why GP's prescribe 'Clavidur' in your coverage?"
    reviews = [
    # Price themed feedback
    [1, "The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
    [2, "The line at each checkout was long, lost a lot of time waiting. Some of the meat prices were very high, can't afford that."],
    [3, "This store is always crowded, long lines at aisles, at checkout register, parking lot. I had to budget 1 hr for waiting in lines and 15 mins on just shopping"],
    [4, "The price of most frozen items are more than the local store. For example, the price of steak is $59.99 whereas the local store is selling for $45. Also, the A/C doesn't work most of the time."],
    [5, "The grocery store had a wide variety of produce. But it was far away from our home by car. So we decided not to go there."],
    [6, "Prices have been increasing consistently at this store, especially for organic products. Considering shopping elsewhere."],
    [7, "Found that dairy products are overpriced compared to other stores in the area. Not happy with the price hike."],
    [8, "Fruits and vegetables are pricey here. I can get the same quality for a lower price at the farmers' market."],
    [9, "Seafood section seems overpriced. The same fish is available for less at the nearby market."],
    [10, "The bakery items are too expensive. Other local bakeries offer better deals."],
    # Delivery themed feedback
    [11, "Ordered groceries online for home delivery, but the delivery was delayed by two days without any notification."],
    [12, "The delivery service is not reliable. Last time my order was missing several items."],
    [13, "Home delivery is convenient, but the delivery charges seem too high compared to other services."],
    [14, "The delivery person was very rude. Unpleasant experience overall with the home delivery service."],
    [15, "I appreciate the home delivery option, but the packaging needs improvement. Found some damaged items upon arrival."],
    [16, "Delivery time slots are never available when I need them. It's frustrating to plan around their limited schedule."],
    [17, "The store's delivery service used to be good, but recently it's been taking longer to receive my groceries."],
    [18, "Ordered fresh produce for delivery but received items that were close to expiry. Not satisfied with the quality control."],
    [19, "Tried the delivery service for the first time, and it was efficient and on time. Good experience overall."],
    [20, "Delivery service was okay, but they missed out on some key items, which was inconvenient."]
    ]
    #[2,"As Discussed with Mr.Ram  Boss,\n\nPSR PDA TMR entry for all the towns should happen on 5th,10th,15th,20th..\n\nBenefits.\n1. Block Reports accessed through  PDA TMR Entr.   It Should reflect in TSE minet.\n2. Datas - Secondary happened in stockist town will be visible.\n3. TSE can understand  better knowledge about the Business & Teritorry & Action planning can be done.\n4.PSR  Efficiency\n5. Business Standards will increase by getting accuracy of data.\n\nThis can be done atleast for main Brands with respect to depot scenario.\n\nThanking You.\n\nRegards,\nH.Dhaamodharun"],
  
    reviews1 = [
        [313, "Quality Gift"],
        [314, "Good company image and statgy"],
        [315, "Focus,regular visit"],
        [316, "E survey , Inputs on clavidur"],
        [317, "Gp s van met all kind of patients"],
        [318, "esurvey gift"],
        [319, "visit"],
        [320, "Regular visit companies"],
        [321, "RELATION SOME ARE E-SURVEY"],
        [322, "Schems and rates"],
        [323, "e survey input"],
        [324, "First linetherapy"],
        [325, "."],
        [326, "No"],
        [327, "Lupin brand Frequent visits"],
        [328, "10"],
        [329, "Focus and e-surveys"],
        [330, "RTI tonsillitis"],
        [331, "gift regular e survey"],
        [332, "Personal Relations"],
        [333, "single focus and e survey"],
        [334, "E-survey Gift Follow up"],
        [335, "FOCUS"],
        [336, "focus single brand e.survey"],
        [337, "1.We have engaged salected customers in PMS activity 2.We have engaged salected customer in input activity"],
        [338, "1 regular visit 2 pms,gift"],
        [339, "1.We have engaged salected customers in PMS activity 2.We have engaged salected customer in input activity"],
        [340, "1) We have engaged selected customer PMS activity. 2) We have engaged selected customers Input activity."],
        [341, "1- We have engaged selected customer in PMS activity. 2- we have engaged selected customer in input activity ."],
        [342, "1 Regular visit 2 Regular Input"],
        [343, "Input/PMS Follow up"],
        [344, "field relations focus"],
        [345, "My regular visit and activity done by the company"],
        [346, "E-survey"],
        [347, "Relationship Continue gift"],
        [348, "Scheme Inputs"],
        [349, "Focused brand remains CLAVIDUR"],
        [350, "E-Survey BDE"],
        [351, "1. Regular visit 2. Input"],
        [352, "good activity plan"],
        [353, "Good bonding Input campaign"],
        [354, "when ever patient required"],
        [355, "when ever patient required"],
        [356, "Regular visits Maintaining good relationship"],
        [357, "found good result"],
        [358, "Regular visit and complement"],
        [359, "Good inputs All under dpc"],
        [360, "They were prescriber"],
        [361, "Fist of marig come to gp"],
        [362, "1. DEMAND 2. FOCUS"],
        [363, "1. Relationship 2. E survey"],
        [364, "they preferred more"],
        [365, "1-E-survey 2- Focus on GP"],
        [366, "1"],
        [367, "100%"],
        [368, "OUR RELATION WITH DRS"],
        [369, "OUR RELATION WITH DRS"],
        [370, "Good quality"],
        [371, "Focused and regular visits"],
        [372, "E survey,inputs"],
        [373, "Esurvey Input"],
        [374, "due to brand of lupin"],
        [375, "They good potential"],
        [376, "Good relations Campaign drs"],
        [377, "E survey Regular visit"],
        [378, "Regular visit"],
        [379, "E survey. With the helps of Mass inputs and sampling."],
        [380, "Quality product"],
        [381, "E Survey factor"],
        [382, "1. E survay 2. My fallow up"],
        [383, "E survey n gifts"],
        [384, "schemes &Activity"],
        [385, "My follow up"],
        [386, "1) E Surveys 2) Gift Campigns conducted"],
        [387, "Inputs ...hard work"],
        [388, "gp's brand"],
        [389, "Quality Regular visit"],
        [390, "Gift"],
        [391, "Tonsillitis phyrangitis"],
        [392, "Regular visits and follow up"],
        [393, "Scheme"],
        [394, "Regular visit Relationship"],
        [395, "Gift utilisation"],
        [396, "E survey and gift"],
        [397, "gift and pms"],
        [398, "FOCUS AND INPUT"],
        [399, "1 PMS ACTIVITY 2 INPUT ACTIVITY"],
        [400, "E survey and gift dr"],
        [401, "Visit and e surcey"],
        [402, "Relation Services"],
        [403, "Focus produt"],
        [404, "1.input"],
        [405, "1 BD 2 GIFT INPUTS"],
        [406, "1regular fallow up 2.e servy"],
        [407, "They preferred amoxcy clavnic"],
        [408, "Gift camping and regular visit"],
        [409, "1-Regular focus 2-activity"],
        [410, "regular visit relation with gp"],
        [411, "only gift"],
        [412, "E survey Gift given by us"],
        [413, "b rand image bd"],
        [414, "1.E survey 2. Input"],
        [415, "1.regular visit 2.bd"],
        [416, "E Survey"],
        [417, "Relationship Lupin brand"],
        [418, "Visit and gratification"],
        [419, "regular inputs"],
        [420, "1 repo with gps 2 help of e Survey"],
        [421, "Price"],
        [422, "Brand image Lupin company Activity"],
        [423, "Rapoo Gift"],
        [424, "REGULAR CALL RATE SCHEME"],
        [425, "REGULAR CALL RATE SCHEME"],
        [426, "Quality Scheme"],
        [427, "1.Regular visit 2.Good relationship with the doctors"],
        [428, "Focused on ds n dds stegntth"],
        [429, "scheme and activity"],
        [430, "GIFTS PROPER VISIT"],
        [431, "Quality product"],
        [432, "PMS Gift campaigns"],
        [433, "1 E survey Engagement"],
        [434, "Dr.s commonly use amoxy clav"],
        [435, "1. relation 2. gift articals"],
        [436, "INPUTS AND FOCUS"],
        [437, "1.Good relationship of doctor"],
        [438, "Quality"],
        [439, "Gift distributed and continually focused on clavidur"],
        [440, "Daily reminder to the dr"],
        [441, "1. E survey 2. scheme"],
        [442, "E survey Gifts"],
        [443, "E SURVEY & GIFTS"],
        [444, "Esurvey Follow up"],
        [445, "FOCUS ON ALL GP DR."],
        [446, "Nice strategy"],
        [447, "1.SERVICE 2.REGULAR VISIT"],
        [448, "1 compani input 2 .bonus"],
        [449, "1- GOOD COMPANY 2-GOOD PRODUCTS"],
        [450, "1. Regular visit 2. Input"],
        [451, "E survey and quality"],
        [452, "Relationship, gifts ."],
        [453, "1. E SURVEY 2."],
        [454, "1. GP CP E SURVEYS 2. Inputs( gift)"],
        [455, "Input Gratification"],
        [456, "Input and regular visit"],
        [457, "company Result"],
        [458, "Esurvey enrolled"],
        [459, "Regular visits and with relationships, and gifts"],
        [460, "1. Regular visit 2. Regular Inputs"],
        [461, "E Survery Gift along with relationship."],
        [462, "WHERE DISPENSING IS NOT A PROB"],
        [463, "My visit Brand recognition"],
        [464, "Ony regular follup."],
        [465, "focus"],
        [466, "GP has good potential GP is gift minded"],
        [467, "our hard work & follow up"],
        [468, "1 INPUT 2 REGULAR VISIT"]
    ]
    reviews1 = [
        [313, "Quality Gift"],
        [314, "Good company image and statgy"],
        [315, "Focus,regular visit"],
        [316, "E survey , Inputs on clavidur"],
        [317, "Gp s van met all kind of patients"],
        [318, "esurvey gift"],
        [319, "visit"],
        [320, "Regular visit companies"],
        [321, "RELATION SOME ARE E-SURVEY"],
        [322, "Schems and rates"],
        [323, "e survey input"],
        [324, "First linetherapy"],
        [325, "."],
        [326, "No"],
        [327, "Lupin brand Frequent visits"],
        [328, "10"],
        [329, "Focus and e-surveys"],
        [330, "RTI tonsillitis"],
        [331, "gift regular e survey"],
        [332, "Personal Relations"],
        [333, "single focus and e survey"],
        [334, "E-survey Gift Follow up"],
        [335, "FOCUS"],
        [336, "focus single brand e.survey"],
        [337, "1.We have engaged salected customers in PMS activity 2.We have engaged salected customer in input activity"],
        [338, "1 regular visit 2 pms,gift"],
        [339, "1.We have engaged salected customers in PMS activity 2.We have engaged salected customer in input activity"],
    ]



   
    customer_name="Safeway Inc"
    question = f"Please provide your feedback abou the store"
    reviews = [
            [1,"The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
            [2,"The line at the each checkout was long, lost lot of time waiting. Some of the meat prices were very high, can't afford that."],
    ]
   
    customer_name="Safeway Inc"
    reviews = [
    # Price themed feedback
        [1, "The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
        [2, "The line at each checkout was long, lost a lot of time waiting. Some of the meat prices were very high, can't afford that."],
        [3, "This store is always crowded, long lines at aisles, at checkout register, parking lot. I had to budget 1 hr for waiting in lines and 15 mins on just shopping"],
        [4, "The price of most frozen items are more than the local store. For example, the price of steak is $59.99 whereas the local store is selling for $45. Also, the A/C doesn't work most of the time."],
        [11, "Ordered groceries online for home delivery, but the delivery was delayed by two days without any notification."],
        [12, "The delivery service is not reliable. Last time my order was missing several items."],
        [13, "Home delivery is convenient, but the delivery charges seem too high compared to other services."],
        [14, "The delivery person was very rude. Unpleasant experience overall with the home delivery service."],
        [15, "Not much comments"],
        [16, "."],
        [17, "No"],
        [18,"N/A"],
        [19, "All good"],
    ] 
   
    
    customer_name="Lupin Pharmacuticals"
    question = f"‌Sales: Please provide all your IDEAS (BIG or SMALL) which will help Reduce Cost or Increase Productivity across functions."        
    reviews = [
            [1,"As Discussed with Mr.Ram  Boss,  PSR PDA TMR entry for all the towns should happen on 5th,10th,15th,20th..  Benefits. 1. Block Reports accessed through  PDA TMR Entr.   It Should reflect in TSE minet. 2. Datas - Secondary happened in stockist town will be visible. 3. TSE can understand  better knowledge about the Business & Teritorry & Action planning can be done. 4.PSR  Efficiency 5. Business Standards will increase by getting accuracy of data.  This can be done atleast for main Brands with respect to depot scenario.  Thanking You.  Regards, H.Dhaamodharun"],
            [2,"1. My idea is that We should focus on Saffola edible oil packaging . We need to improve the quality of pouches.Its very poor compared to other players in market. This improvisation will help us improve our market share. 2. In Oats Flavour packs are only .43gms. The market is demanding for bigger size packets. I do understand the issue of maintaining the quality in bigger packs. But a innovation in this would help us gain the early bird advantage. 3.I have an IDEA , we can come up with CUP Oats(masala) like the cup Noodles. This would help us enter Eatery channels.  4.We can bring Bulk Breaking for upcountry Midas DDs, This would help us bill slow moving SKUs to tier 2 markets. This would create interest in upcountry DDs to focus on range selling of slow moving SKUs.  5. We should again seriously implement or focus on VMI generated orders for upcountry DDs. This would bring back confidence in our Systems for our DDs.  6. We should focus on educating the retailers, DSRs and end consumers about the Blend, advantages of kardi and merits of using Saffola. Which I think offlate we are not doing .  Thank You  Smithun Chennai Metro TSE Metro 9176686642."],
            [3,"to increase the productivty, a dsr should work from both the way in one beat, one week from one way and one from other way, as many outlet get close or get out of cash for the payment of old bill"],
            [4,"to increase the productivty, a dsr should work from both the way in one beat, one week from one way and one from other way, as many outlet get close or get out of cash for the payment of old bill"],
            [5,"1. Online shopping site for sales of Marico products at discounted price for Customers and Retailers, with a free doorstep delivery. 2.  Online games for promotion of Youth portfolio range     a) Set Wet gaming Zone    b) Tie up with X Box live games 3.  Explore new set of outlets for Saffola Oates range in Bakery . 4.  Promote Livon Hair Gain & Therapie  Brand with support of BPOs meeting Dermatologists to increase sales in chemist outlets.   5. Promotion of brands like Saffola Oates and Edible oils through Gym Instructors"],
            [6,"Reducing Cost :In Marico we reimburse DD commission to stockist as part of redistribution expense, at present it ranges from .05% to .25%, and sitting in present banking scenerio, most of the bank participate online banking/RTGS facility with minimum charges, and evaluation can be done on this and  it definitely will reduce the charges that is presently borne by Marico."],
            [7,"Month Target should be uploaded by 2nd of every month in the morning only so that TSO will seat with DSR in JC Meet along with Actual target. If DSR have some issue with more target, we can tell him month supported scheme for specific brands and logic behind target."],
            [8,"1. My idea is that we have to appoint more CFA or Under CFA unit so that we can decrease the cost of transportation and give service batter in urban towns.  2. We should tie-up with specialty outlets  (like Westside , Petaloons, Shopper stop) for our new brands ie..Set wet Deos and Hair Gain because of this type of product is already available there."],
            [9,"- L&D - I case of cap breakage could we provide depot with extra caps to reduce the L&D Cost and Salvage SKUs with 100%  Stock in it."],
            [10,"As you know that POP plays a vital role in our sytem . we need to give some Code no to all POP . we should dispatch the POP with SAP code to DD/SD. The same needs to be tracked by TSO/TSE through MI_NET."],
            [11,"We should make launch a type of comby pack in our market. Just like as Saffola Gold 1pc+ Saffola Rise 1 pc+ H&C 1pc and with a Slow move item (free 50 ml Santi Amla) and distribut it in A class outlet."],
            [12,"I have an IDEA regarding customer awareness, if all the New Product lunch Detail, Offers Detail, Points Detail, Target Vs Achievement Detail, Anniversary and Birthday wishes provide through SMS to top 20% customer who contribute 80% of business in short UNNATI, MERA and BANDHAN parties. Then it can help us to minimise paper work and increase customer relation."],
            [13,"Provide best IT instrument to User , Like Laptop or Desktop which generally changes(In case of Laptop)after 4 year &(In case of desktop) 6 to 7 year respectively, which is not workable after this much long time duration,So my Idea is to re validate this time period. so that User performance not hampered."],
            [14,"We can arrange for a 'Marico Loyalty Coupon' in open format outlets. For every Rs. 1000 worth of purchase of Marico Ltd products, a Rs 20 Marico Ltd coupon can be given to consumers which they can redeem on their next Marico purchase. This will serve the dual purpose of increasing sales and establishing corporate brand equity."],
            [15,"Proper way in implementing Stockist PDP at SD level will reduce Freight charges."],
            [16,"If we have corporate mobile connection plan for all company person will also reduce mobile bill cost."],
            [17,"1st idea----- TO increase the produtivity  Dsr  shuold  sincere on every sku  of marico .                 2nd  idea-----DSR  should  make  habbit  to  open new  outlet & also  care  on the weidth  of  the BEAT. 3rd idea--  company  has given certificate  to  best WMP  , DHOOM,KEY OUTLET  on every   year in the   same   way  every year  given BEST  Distributors award &  certificate"],
            [18,"1st idea----- TO increase the produtivity  Dsr  shuold  sincere on every sku  of marico .                 2nd  idea-----DSR  should  make  habbit  to  open new  outlet & also  care  on the weidth  of  the BEAT. 3rd idea--  company  has given certificate  to  best WMP  , DHOOM,KEY OUTLET  on every   year in the   same   way  every year  given BEST  Distributors award &  certificate"],
            [19,"1st idea----- TO increase the produtivity  Dsr  shuold  sincere on every sku  of marico .                 2nd  idea-----DSR  should  make  habbit  to  open new  outlet & also  care  on the weidth  of  the BEAT. 3rd idea--  company  has given certificate  to  best WMP  , DHOOM,KEY OUTLET  on every   year in the   same   way  every year  given BEST  Distributors award &  certificate"],
            [20,"1. All employees should have corporate level telecommunication plans. 2. All old Performing DSRs/PSRs/ISRs  should be promoted and old non performing DSRs/ PSRs/ISRs  should be replaced."],
            [21,"In Mi-net there is A Report/Tracker Availble Daily DSR Wise/ Bpd Report ,If  In Same Format Daily DSR Wise/Bpm and Pcno Vol  ReporT Come. It Help Us Across Level For Review Day to Day Business Updates In One Single  Sheet Like Bpd Report,Please Start Give this Report Asap,"],
            [22,"1. Bandhan parties are selected through a Conditional  BPM. It Should Be Review in every cover. 2. Those parities are failed to achieve their Minimum conditional BPM in a cover, his catdom chque will be withdrawn."],
            [23,"1.Should increase reach in rural penitration. 2.Should have micro coverage in town of population 1000. A pocket whole sale as micro stockist.  3. we can also deploy cycle salesman who can cover the interior markets & take stocks from near by stockist on commission basis on sales , which in turn can increase reach. 4 Can use display of Rs 100 a month to main outlet of the small town to give visiblity of brands."],
            [24,"Our PSRs DSRs and ISRs at many times do not get updates on where they are in terms of  incentive earning status.Now they all have PDAs. Can they get updates on PDAs on 10th, 20th and 25th of every month"],
            [25,"SEPERATE DISTRIBUTORS FOR PERSONAL N HEALTH CARE PRODUCTS AS THER ARE NO OF PRODUCTS N SKU TO BE HANDLED BY A DSR.IT WILL INCREASE FOCUS OVER SALES ,BETTER COVERAGE AND LESS FINANCIAL DEPENDABILITY ON ONE DISTRIBUTOR IN METRO TOWNS WHOSE TURNOVER IS MORE THAN RS 80 LACS. IN METRO OR A CLASS TOWNS WHILE GOING FO NEW DISTRIBUTOR WE SHOULD UPGRADE 2ND LINE FMCG CO'S DISTRIBUTOR RATHE THAN GOING FOR TOP FMCG CO'S DISTRIBUTOR WHOSE TURNOVER IS MORE THAN OUR BUSINESS.SUCH DISTRIBUTORS ARE NOT DEPENDENT ON OUR BUSINESS. IN METROS OR A CLASS TOWNS DISTRIBUTORS SHOULD BE OF SMALLER SIZE AREAS FOR BETTER COVERAGE,FASTER DELIVERY N GROWTH."],
            [26,"IN DSR PDA SRN VALUE CALCULATED AS GROSS VALUE OF THE PRODUCT, THIS WILL TAKE THE OPERATOR TO SPEND TIME ON BILLING EVEN IN DDR SCENARIO. IF IT COMES AS NET LANDING RATE OR SALVAGE RATE HELP TO REDUCE EXCESS AMOUNT PASSED TO THE RETAILER AND KEEP AS 100% SRN BY PDA"],
            [27,"IN DDR ALL THE SKUs MAPPED IN RUNNING CODE NOTHING WILL BE LAPSED, THIS WILL GIVE 100% FILL RATE FOR STOCK AVAILABLE BRANDS, THE SAME GET INTRODUCED IN MIDAS THE DSR WILL NOT LOOSE THE SECONDARY BECAUSE OF NEW SKU/DUPLICATE SKU IN PDA,NO NEED TO CHECK AND BILL AS PER THE PENDING ORDER REPORT(TIME SAVING AS WELL AS MINIMIZE THE SALE LOSS)"],
            [28,"WHY CANT WE FOCUS ON  MALE GROOMING(AFTERSHOWER) THE MONOPOLIST OUT OF RACE WE HAVE CHANCE TO GRASP THE SHARE, BRYLCREAM MAY COMEBACK ANY MONTH, WE CAN PLAN an MM ACTIVITY FOR MALEGROOMING TO CAPTURE THE MS TO USE THIS OPPURTUNITY."],
            [29,"NOW OLIVE OIL MARKET IS GOING UP WHY CANT WE BRING BACK SAFFOLA OLIVE OIL, IN TN KARDIA THE BRAND FROM KALEESWARY REFINARIES DOING WONDERS WHY CANT MARICO DIDN't.(KARDIA IS A BLEND OF OLIVE AND CORN targetting KOCO)"],
            [30,"increase productivity- Today, we run channel/slab scheme in urban midas towns.my suggestion is we should run QPS(QUANTITY PURCHASE SCHEME) in wholesale channel (atleast for semi wholeseller) instead of slab scheme.because in slab scheme the wholesaler has to purchase in one bill.but for semi whole seller it could be a difficult to go for high slab.if we map QPS system in MIDAS,then we can map scheme as like the scheme will deduct in the last bill at fourth week wholesale beat.i think semi wholeseller's buying capacity & buying frequency can be make high by this system."],
            [31,"increase productivity-in PARAS segments  we should run net rate scheme or percentage scheme because in DEO segments each & every competitors run flat/percentage scheme rather than litter scheme.so it is difficult for DSRs to make retailers understand rate or scheme of DEO.sometimes it create difficulties & confusion for both side.if we mapped flat/percentage scheme in MIDAS for PARAS segments,then bill could be understandable for retailers & dsr also."],
            [32,"can we use TAB instead of samsung galaxy for PDA usage.as TAB is more user friendly than mobile.big screen & more features can be easily handled by DSRs.in price factor also,we can get TAB between 10 k to 12 k.so in price factor point view also it could be better than SAMSUNG GALAXY."],
            [33,"we should activate AUTOMATIC UPDATE of stock in PDA.for example,1 DD has 50 cases of pcno 175 ft stock in the morning,when DSRs taking orders from the market it should automatically reduce from 'SIH' quantity.so that DSR can get actual stock from time to time in the market & no order will be deducted when 'ORDER TO BILL CONVERSION' has done."],
            [34,"can we pass on the amount of display scheme like MERA DISPLAY AMOUNT,CATDOM DISPLAY AMOUNT OR ANY OTHER DISPLAY AMOUNT more quickly & frequently to the market after ending of  phase.sometimes it comes so late that we & DSRs faces lots of problems in the market."],
            [35,"POS utilization can be improved. External Agency for effective utilization of POS. Reusable POS if possible will help."],
            [36,"Mandays and BPD details to appear in DSR JC to improve productivity."],
            [37,"Replacing TSE laptops with Midas inbuilt TAB for portability and convinience of usage"],
            [39,"In PAHHO FOR AWARENESS AND BRAND STABILITY PUT ADD IN ALL LOCAL NEWS PAPER MINIMUM 15 DAYS FOR ATLEAST 4 MONTHS."],
            [40,"In PAHHO FOR AWARENESS AND BRAND STABILITY PUT ADD IN ALL LOCAL NEWS PAPER MINIMUM 15 DAYS FOR ATLEAST 4 MONTHS."],
            [41,"In PAHHO FOR AWARENESS AND BRAND STABILITY PUT ADD IN ALL LOCAL NEWS PAPER MINIMUM 15 DAYS FOR ATLEAST 4 MONTHS."],
            [42,"BODY LOTIN 20 ML SHOULD COMES IN JAR PACKAGING AND FOR MAXIMISE MARKET SHARE IT SHOULD BE START IN THE MONTH OF SEPTEMBER."],
            [43,"As per our company policy, company do not provide Laptops to TSOs. But in our system, one cannot work without a Laptop  My suggestion is that we should encourage TSOs to buy & use their Personal laptop. A  Purchase bill has to be submitted by TSOs to company. Company will reimburse the amount in EMI of 24 Months, starting from first month of Purchase. Only condition is TSO has to stay with us for at least one year to actually get this benefit  Now, If an employee leaves company before completion of 1 year, the amount already given as EMI to employee to be deducted in his F & F settlement.  If an employee leaves between 1 yr and 2 yr- He will get EMI till that period.  By this way, Company will not have to bear onetime cost & same time company will give benefit of this scheme to those employee only, who stays  with us for more that 1 year."],
            [44,"1. New Programme shall be introduced for Pure Cosmatic wholesale outlets as the bill amount vs points achived in current scenario is higher for Non Paras WMPs. For eg For a 1 case of Livon SP a  cosmaetic Wholesaler will incur avg rs.7000 and will earn 6 points whereas if WMP buys PCNO Rigid fof same amount he will earn nearly 29 points. The point structure for Cricket products shall be changed so as match with regular WMP."],
            [45,"Have centralized corporate tie-ups with travel companies and hotels across location covering all bands of employees , will result in good saving in our travel and stay expenses."],
            [46,"Rationalizing quantity of POS with short life and investing in more sustainable and long term POS  eg. Poster Vs Shop Branding"],
            [47,"Incorporating learning's  form local players in our strategy eg. In case of Ayurvedic hair oil space print seems to be most credible media and local players like SESA , KESH KING invest in Print so we investing in TV may not work."],
            [48,"Cost of carrying inventory can be reduced by having some level of flexibility in backend production. Need to integrate the system to capture sales trend and accordingly decide on norm of inventory at which Stop / Start alert can be raised to plant. To have flexibility at 3P's we can modify their cost structure to slab rate on p.a basis."],
            [49,"We have a lot of seasonal products in our kitty now. So before the season starts and loading of competition seasonal products happen we should call a meet of all the HTP outlets and load them to the fullest. For Eg in case of body lotion we should call all the HTP outlets in Mid Sept and load them to the max of their potential. The same should happen for Angol brands in Oct and Deos in Jan End."],
            [50,"1. Regarding Pcno Blister we are not giving sufficient stock to retail & wholesale because of stock issue. As we Know the production issue of blister. My IDEA LAUNCH OF 1/- SACHET PACK This will reduce the production cost and proper stock fullfillment in market place.  2. Regarding Sales Offload in Distributor point all Jc's of TSO should happen on 30th or 31st of every month --- this leads proper offload of all DSR's team --- and much productivity of every month.  thank you"]
    ]

    customer_name="Safeway Inc"
    reviews = [
    # all themed feedback
        [1, "The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
        [2, "The line at each checkout was long, lost a lot of time waiting. Some of the meat prices were very high, can't afford that."],
        [3, "This store is always crowded, long lines at aisles, at checkout register, parking lot. I had to budget 1 hr for waiting in lines and 15 mins on just shopping"],
        [4, "The price of most frozen items are more than the local store. For example, the price of steak is $59.99 whereas the local store is selling for $45. Also, the A/C doesn't work most of the time."],
        [11, "Ordered groceries online for home delivery, but the delivery was delayed by two days without any notification."],
        [12, "The delivery service is not reliable. Last time my order was missing several items."],
        [13, "Home delivery is convenient, but the delivery charges seem too high compared to other services."],
        [14, "The delivery person was very rude. Unpleasant experience overall with the home delivery service."],
        [15, "Not much comments"],
        [16, "."],
        [17, "No"],
        [18, "N/A"],
        [19, "All good"],
        [20, "Parking was not there.  i had to park on the street"],
        [21, "More parking spots please. also increase more sleection of fresh produce"],
        [22, "Parking always has ben issue here, so i stopped going to this location. Unpleasant experience overall with the home delivery service."],
        [25, "Parking is always a hassle, especially during peak hours. It's almost impossible to find a spot."],
        [26, "Found the prices here a bit too steep for my budget. Had to limit my purchases."],
        [27, "The selection of fresh produce is quite limited. Wish there were more organic options."],
        [28, "Disappointed with the meat quality. It didn't seem very fresh and had a strange smell."],
        [29, "Restrooms were out of order during my last visit, which was quite inconvenient."],
        [30, "Overall, the store experience was underwhelming. The aisles were cluttered and disorganized."],
        [31, "Had trouble finding a parking spot. Too crowded and poorly managed."],
        [32, "Prices have gone up recently, making it difficult to shop here regularly."],
        [33, "Wish they had a wider variety of fruits and vegetables. The selection is too basic."],
        [34, "The meat section had a limited choice, and the quality was just average."],
        [35, "The restrooms were dirty and lacked basic supplies. Not a pleasant experience."],
        [36, "The store layout is confusing, and the staff seemed disinterested in helping."],
        [37, "Parking was a nightmare. I had to circle around multiple times to find a spot."],
        [38, "I found the same items cheaper at another store. Prices here are too high."],
        [39, "There's a lack of variety in the produce section. It's the same old stuff every time."],
        [40, "The meat looked unappetizing and old. I decided not to buy any."],
        [41, "Had a bad experience with the restrooms. They were not maintained well."],
        [42, "The store was chaotic, and it was hard to move around with the shopping cart."],
        [43, "Struggled with parking. All spots were taken, and no one was directing the traffic."],
        [44, "The price hike on dairy products was unexpected. Might have to shop elsewhere."],
        [45, "The produce section needs improvement. Not much variety or freshness."],
        [46, "Meat selection was disappointing. The cuts weren't great and looked old."],
        [47, "Restrooms need serious attention. They were smelly and unclean."],
        [48, "The store was too crowded, and the shelves were disorganized. Not a great shopping experience."],
        [49, "Parking is a major issue here. Took me 20 minutes to find a spot."],
        [50, "I noticed a significant price increase in basic groceries. It's becoming unaffordable."]           
    ]

    return customer_name, question_id, question, reviews

if __name__ == "__main__":
    # sample cli call 
    # py prep_data_llm.py unit_test 100 1234 10
    # py prep_data_llm.py db 100 1234 10
    category=sys.argv[1]
    pipeline_job_id=sys.argv[2]
    question_id=sys.argv[3]
    record_count=sys.argv[4]

    os.environ['PIPELINE_JOB_ID'] = str(pipeline_job_id)

    if record_count is None:
        record_count = 50

    if len(sys.argv) > 1:
        # The first argument in sys.argv[1:] will be your parameter
        main(category, pipeline_job_id, question_id, record_count)
    else:
        print_debug("No parameter provided. Running unit test with default settings.")
        unit_test(None)


"""
def summarize_nested_array(nested_array, bad_data_array):
    unique_answer_ids = set()
    split_sentences_count = 0
    unique_group_label_auto = set()
    unique_master_insight = set()
    unique_insight = set()
    total_feedback_score = 0
    master_insight_details_with_split_sentences = {}  # New dictionary

    # Track the number of answer_ids and split_sentences for each master_insight
    master_insight_stats = {}

    # Iterate through nested array to collect data
    for array in nested_array:
        for record in array:
            answer_id = record["answer_id"]
            master_insight = record.get("master_insight", "Not Available")
            unique_answer_ids.add(answer_id)
            split_sentences_count += 1
            unique_group_label_auto = record.get("group_label_auto", "Not Available")
            unique_master_insight.add(master_insight)
            insight = record.get("insight", "Not Available")
            split_sentence = record.get("split_sentence", "")  # Assuming there is a key 'split_sentence'
            unique_insight.add(insight)
            total_feedback_score += record.get("answer_score", 0)

            # Update stats for master_insight
            if master_insight not in master_insight_stats:
                master_insight_stats[master_insight] = {"answer_ids": set(), "split_count": 0}
                master_insight_details_with_split_sentences[master_insight] = []  # Initialize list for each master_insight
            master_insight_stats[master_insight]["answer_ids"].add(answer_id)
            master_insight_stats[master_insight]["split_count"] += 1
            master_insight_details_with_split_sentences[master_insight].append(split_sentence)

    # Calculating summary stats
    bad_records_count = len(bad_data_array)
    unique_answer_ids_count = bad_records_count + len(unique_answer_ids)
    avg_feedback_score = total_feedback_score / split_sentences_count if split_sentences_count > 0 else 0
    avg_split_sentences_per_answer = split_sentences_count /  len(unique_answer_ids) if unique_answer_ids else 0
    bad_records_percentage = (bad_records_count / unique_answer_ids_count * 100) if unique_answer_ids else 0
    # Formatting master_insight stats
    master_insight_summary = {mi: (len(stats["answer_ids"]), stats["split_count"]) for mi, stats in master_insight_stats.items()}

    if len(unique_master_insight) > 0:
        count_of_split_sentence_per_master_insight = round(split_sentences_count / len(unique_master_insight), 2)
    else:
        count_of_split_sentence_per_master_insight = 0

    summary = {
        "total_number_of_unique_answer_ids": unique_answer_ids_count,
        "total_number_of_split_sentences": split_sentences_count,
        "average_number_of_split_sentences_per_answer": avg_split_sentences_per_answer,
        "total_number_of_unique_group_label_auto": len(unique_group_label_auto),
        "total_number_of_unique_master_insight": len(unique_master_insight),
        "total_number_of_unique_insight": len(unique_insight),
        "average_feedback_score": avg_feedback_score,
        "count_of_split_senence_per_master_insight": count_of_split_sentence_per_master_insight,
        "percentage_group_label_auto_per_answer_id": f"{len(unique_group_label_auto) / unique_answer_ids_count * 100:.2f}%",
        "number_of_bad_records": bad_records_count,
        "percentage_of_bad_records": f"{bad_records_percentage:.2f}%",
        "list_of_unique_master_insights": list(unique_master_insight),
        "master_insight_details_with_ans_id_and_split_counts": master_insight_summary,
        "master_insight_details_with_split_sentences": master_insight_details_with_split_sentences
    }

    return json.dumps(summary, indent=2)
def summarize_nested_array(nested_array, bad_data_array):
    unique_answer_ids = set()
    split_sentences_count = 0
    unique_group_label_auto = set()
    unique_master_insight = set()
    unique_master_insight_agg = set()
    unique_insight = set()
    total_feedback_score = 0

    # Dictionaries for master_insight stats and details
    master_insight_stats = {}
    master_insight_agg_mapping = {}
    master_insight_details_with_split_sentences = {}  # New dictionary

    # Iterate through nested array to collect data
    for array in nested_array:
        for record in array:
            # Common data extraction
            answer_id = record["answer_id"]
            unique_answer_ids.add(answer_id)
            split_sentences_count += 1
            group_label_auto = record.get("group_label_auto", "Not Available")
            unique_group_label_auto.add(group_label_auto)
            insight = record.get("insight", "Not Available")
            split_sentence = record.get("split_sentence", "")  # Assuming there is a key 'split_sentence'
            unique_insight.add(insight)
            total_feedback_score += record.get("answer_score", 0)

            # Master insight data extraction
            master_insight = record.get("master_insight", "Not Available")
            unique_master_insight.add(master_insight)
            
            # Master insight aggregated data extraction
            master_insight_agg = record.get("master_insight_agg", "Not Available")
            unique_master_insight_agg.add(master_insight_agg)

            # Mapping master insights to aggregated master insights
            if master_insight_agg not in master_insight_agg_mapping:
                master_insight_agg_mapping[master_insight_agg] = set()
            master_insight_agg_mapping[master_insight_agg].add(master_insight)

            # Update stats for master_insight
            if master_insight not in master_insight_stats:
                master_insight_stats[master_insight] = {"answer_ids": set(), "split_count": 0}
            master_insight_stats[master_insight]["answer_ids"].add(answer_id)
            master_insight_stats[master_insight]["split_count"] += 1
            master_insight_details_with_split_sentences[master_insight].append(split_sentence)
       
    # Calculating summary stats
    bad_records_count = len(bad_data_array)
    unique_answer_ids_count = bad_records_count + len(unique_answer_ids)
    avg_feedback_score = total_feedback_score / split_sentences_count if split_sentences_count > 0 else 0
    avg_split_sentences_per_answer = split_sentences_count / len(unique_answer_ids) if unique_answer_ids else 0
    bad_records_percentage = (bad_records_count / unique_answer_ids_count * 100) if unique_answer_ids_count > 0 else 0

    # Formatting master_insight stats
    master_insight_summary = {mi: (len(stats["answer_ids"]), stats["split_count"]) for mi, stats in master_insight_stats.items()}
    count_of_split_sentence_per_master_insight = round(split_sentences_count / len(unique_master_insight), 2) if len(unique_master_insight) > 0 else 0
    count_of_split_sentence_per_master_insight_agg = round(split_sentences_count / len(unique_master_insight_agg), 2) if len(unique_master_insight_agg) > 0 else 0

    # Convert master_insight_agg_mapping to desired format
    master_insight_agg_mapping_format = {agg: list(insights) for agg, insights in master_insight_agg_mapping.items()}

    summary = {
        "total_number_of_unique_answer_ids": unique_answer_ids_count,
        "total_number_of_split_sentences": split_sentences_count,
        "average_number_of_split_sentences_per_answer": avg_split_sentences_per_answer,
        "total_number_of_unique_group_label_auto": len(unique_group_label_auto),
        "total_number_of_unique_master_insight": len(unique_master_insight),
        "total_number_of_unique_master_insight_agg": len(master_insight_agg),
        "total_number_of_unique_insight": len(unique_insight),
        "average_feedback_score": avg_feedback_score,
        "count_of_split_senence_per_master_insight": count_of_split_sentence_per_master_insight,
        "count_of_split_senence_per_master_insight_agg": count_of_split_sentence_per_master_insight_agg,
        "percentage_group_label_auto_per_answer_id": f"{len(unique_group_label_auto) / unique_answer_ids_count * 100:.2f}%" if unique_answer_ids_count > 0 else "0.00%",
        "number_of_bad_records": bad_records_count,
        "percentage_of_bad_records": f"{bad_records_percentage:.2f}%",
        "list_of_unique_master_insights": list(unique_master_insight),
        "list_of_unique_master_insight_agg": list(unique_master_insight_agg),
        "master_insight_details_with_ans_id_and_split_counts": master_insight_summary,
        "master_insight_details_with_split_sentences": master_insight_details_with_split_sentences,
        "master_insight_agg_mapping_master_insight": master_insight_agg_mapping_format
    }


    return json.dumps(summary, indent=2)

"""