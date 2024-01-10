"""
usecase : 
100 question: 10 labels.  hit generate:  it put labels in 20.   
user updates it and then hit generate.  it comes back with another 10. 

1. test with answers and integrate with mysql
2. integrate with question 
3. 100 answers, 20 labels with answers. 
4. copy it from mysql to vector DB. it will overwrite.
5. generate label.  it will match 80 answrs and pull 3 labels 
6. it will update mysql DB

"""

import random
import time
import sys
import os
import json
import numpy as np
import pickle
from utils import print_debug, exec_sql
import utils 
import json
import openai
from sentence_transformers import SentenceTransformer, util
ST_SAVE = True  # Set this to True to use the saved ST training model

openai.api_key = "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"

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    IndexType
)
from sentence_transformers import SentenceTransformer

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
print(sys.executable)

import pymysql
model_name="sentence-transformers/all-MiniLM-L6-v2"
collection_name_label = "answers"
top_k=5 

#milvus search results.  i.e for every sentence, how many max search results milvus search should return
# setting this value lower may miss labels if training data has lot of records per label.
search_count_limit = 35 

def init_vector_db():
    
    # Connect to Milvus server
    print_debug("Start connecting to Milvus")
    try:
        connections.connect("default", host="localhost", port="19530")
        print_debug("Successfully connected to Milvus!")
    except Exception as e:
        print_debug("Failed to connect to Milvus:", e)
        sys.exit(1)
   
def get_latest_version(question_id, next_version=False):
    """
    Get the latest version for saving a trained model.
    Args: question_id (int): The ID of the question.
    Returns: str: The latest version for saving the model.
    """
    base_path = os.path.abspath('../../st_models')
    question_path = os.path.join(base_path, str(question_id))

    if not os.path.exists(question_path):
        if next_version:
            os.makedirs(question_path)
            version_path = os.path.join(question_path, 'v1')
        else:
            return None
    else:
        version_folders = [folder for folder in os.listdir(question_path) if folder.startswith('v')]
        if not version_folders:
            return None
        else:
            latest_version = max([int(folder[1:]) for folder in version_folders])
            if next_version:
                new_version = latest_version + 1
            else:
                new_version = latest_version
            version_path = os.path.join(question_path, f'v{new_version}')
            if next_version:
                os.makedirs(version_path)

    return version_path

def get_labels(data, is_DB=False, pipeline_job_id=0, question_id=0):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer(model_name)
    collection_name = collection_name_label + '_' + str(question_id)
    print_debug(data)
    # If data is retrieved from the database, extract the raw text from the input
    if is_DB:
        cleaned_data = [(item[0], json.loads(item[1])[0]) for item in data]
        print_debug("cleaned data")
        print_debug(cleaned_data)
        delete_query = f"DELETE FROM ai.ai_label_vectordb WHERE pipeline_job_id={pipeline_job_id}"
        print_debug("cleaned up ai.ai_label_vectordb at start")
        exec_sql(delete_query)
    else:
        cleaned_data = data

    # Load the collection using the existing schema
    search_collection = Collection(collection_name)
    search_collection.load()
    print_debug(f"Row count of the answer collection: {search_collection.num_entities}")
    # Search based on vector similarity
    print_debug("Start searching based on vector similarity")
    search_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",  # Cosine similarity
        "params": {"nprobe": 16},
    }
    # Normalize the embeddings
    embeddings = model.encode([new_answer[1] for new_answer in cleaned_data]).tolist()
    # Perform a search with the normalized embeddings
    search_result = search_collection.search(embeddings, "embeddings", search_params, limit=search_count_limit,
                                             output_fields=["pk", "answer", "insight_labels"])
    # Initialize the list for similar questions
    all_new_answers_with_labels = []
    counter = 0
    # Iterate over each new answer
    for new_answer, result in zip(data, search_result):
        new_answer_id = new_answer[0]
        new_answer_text = new_answer[1]
        print_debug("answer:" + new_answer_text + " search results count:" + str(len(result)))
        print_debug(result)
        print_debug("******")
        # Initialize the list for similar questions for the current new answer
        similar_questions = []
        output_records = []
        # Deduplication logic based on least distance
        label_score_dict = {}
        for rank, hit in enumerate(result, start=1):
            matched_answer_label = hit.entity.insight_labels if hit.entity.insight_labels else None
            score = hit.distance
            if matched_answer_label not in label_score_dict or score < label_score_dict[matched_answer_label]:
                label_score_dict[matched_answer_label] = score
            similar_question = {
                'label': matched_answer_label,
                'distance': score,
                'rank': rank,  # Changed to use the actual rank value
                'matched_answer': hit.entity.answer
            }
            similar_questions.append(similar_question)
        # Sort the labels based on distance (in ascending order) , reverse = True for distance descending order
        sorted_labels = sorted(label_score_dict.items(), key=lambda x: x[1], reverse=False)
        # Apply cutoff threshold of 0.8
        #sorted_labels = [(label, distance) for label, distance in sorted_labels if distance <= 0.8]
        # Extract the unique labels based on least distance
        unique_labels = [label for label, _ in sorted_labels[:top_k]]
        # Convert the unique_labels list to a string and escape any single quotes inside the labels
        unique_labels_str = json.dumps(unique_labels).replace("'", "\\'")
        # Append the unique labels to the output list, maintaining the order of least distance first
        output_records = [similar_question for similar_question in similar_questions if similar_question['label'] in unique_labels]
        
        # Create a subset of output_records containing only the top labels from unique_labels
        from operator import itemgetter
        # Create a subset of output_records containing only the top labels from unique_labels
        output_details_top = []
        for label in unique_labels:
            records_for_label = [record for record in output_records if record['label'] == label]
            
            # Get the record with the minimum distance for the current label
            min_distance_record = min(records_for_label, key=itemgetter('distance'))
            output_details_top.append(min_distance_record)
        print_debug(unique_labels_str)
        print_debug(output_details_top)
        output_details_top_json = json.dumps(output_details_top)
        # If is_DB is True, perform DB operations to save the results
        if is_DB:
            print_debug("Doing DB operation, updating mysql db with labels")
            # Construct the SQL query to save the record
            question_id, answer_id = new_answer_id.split('_')
            json_labels = json.dumps(output_records, separators=(',', ':'))  # Convert to JSON string
        
            insert_query = "INSERT INTO ai.ai_label_vectordb (pipeline_job_id,question_id, answer_id, \
                question_text, answer_text, pipeline_output, output_details, output_details_all) \
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)" 
            values = (pipeline_job_id, question_id, answer_id, 'N/A', new_answer_text, unique_labels_str, output_details_top_json, json_labels)
            # delete existing records
            delete_query = f"DELETE FROM ai_label_vectordb WHERE answer_id={answer_id}"
            print_debug(delete_query)
            exec_sql(delete_query)
            exec_sql(insert_query, values)
            # update crm db
            # update the CRM ai db with the labels
            update_query_crm = f"UPDATE ai_pipeline.ai_label_vectordb a join ai.ai_label_vectordb b \
                ON a.pipeline_job_id=b.pipeline_job_id \
                AND a.answer_id=b.answer_id \
                SET a.pipeline_output= JSON_QUOTE(b.pipeline_output), \
                    a.output_details=b.output_details, \
                    a.output_details_all=b.output_details_all, \
                    a.job_status='Processed' \
                WHERE a.pipeline_job_id={pipeline_job_id}"
            exec_sql(update_query_crm)
            counter += 1
            progress_sql = f"""UPDATE ai.ai_pipeline_jobs 
                SET job_progress_level = ( 
                SELECT IFNULL(SUM(CASE WHEN job_status='Processed' THEN 1 ELSE 0 END) / COUNT(1), 0)
                FROM ai.ai_label_vectordb
                WHERE pipeline_job_id = {pipeline_job_id}
                )
                WHERE pipeline_job_id = {pipeline_job_id}
            """
            if counter % 30 == 0:
                print_debug("******** counter  = " + str(counter) + "**********")
                exec_sql(progress_sql)
                import time
                print_debug("sleeping for 2 secs")
                time.sleep(2)  # Sleep for 5 seconds
        new_answer_with_labels = [new_answer_id, new_answer_text, unique_labels, output_records]
        all_new_answers_with_labels.append(new_answer_with_labels)
    # mark progress to be 100% at end of the process    
    #exec_sql(progress_sql)
    # Convert the result to JSON
    json_result = json.dumps(all_new_answers_with_labels, indent=4)
    print_debug(json_result)
    print_debug("Number of records " + str(counter))

# we  move the record stats from schduled to Completed skipping inprocess
def update_data_for_search(pipeline_job_id):
    sql_ai_user_data = f"UPDATE ai_pipeline.ai_label_vectordb SET job_status='Processed' WHERE pipeline_job_id={pipeline_job_id}"
    exec_sql(sql_ai_user_data)
    
def get_data_for_search(pipeline_job_id=0):
    # Retrieve the search data array
    sql = f"SELECT concat(question_id, '_',answer_id) as answer_id,pipeline_input \
            FROM  ai_pipeline.ai_label_vectordb  \
            WHERE job_status='Scheduled' \
            AND pipeline_job_id={pipeline_job_id}"
    
    status, result = exec_sql(sql)
    
    search_data = []
    for row in result:
        answer_id = row[0]
        answer_text = row[1]
        search_data.append([answer_id, answer_text])
    
    print_debug("number of raw answers to get labels for: " + str(len(search_data)))

    return search_data

def main(operation, pipeline_job_id=0, question_id=0):
    # Initialize the vector DB
        
    init_vector_db()
    top_k=5
    if operation == "get_label":
        # Get data for search operation
        search_data = get_data_for_search(pipeline_job_id)
        print(search_data)
        print("\==============")
        # Get labels for search data
        label_result = get_labels(search_data,True,pipeline_job_id, question_id)
        print("Label Result:")
        print(label_result)
        #update_data_for_search(pipeline_job_id)

    elif operation == "unit_test_search":
        unit_test_search()
    else:
        print("Invalid operation. Please specify 'save_trng' or 'get_label' or unit_test_save or unit_test_search.")

# unit test
def unit_test_search():
    search_data = [
        ("1_21", "This pizza was absolutely delicious"),
        ("1_22", "I did not like the crust of the pizza"),
        ("1_23", "crust was good but toppings were stale"),
    ]
    search_data = [
        ("1_21", "1. No knurling, 2. Rate, 3. Company name not mentioned"),
        ("1_22", "there is no branding of on product channel like other channels, the price is marginally higher when compared to local channels"),
        ("1_23", "Price 2. Not having company brand name  3. Not big difference in quality")
    ]

    search_data = [
        ("1_25", "its price\nits over all design ts look very low in same categaoery if i rpresent local .45 material\nand that is also cheaper than expert"),
        ("1_26", "2. BRAND name is missing\n3. low width\n4. no knurling"),
        ("1_27", "1) COMPANY BRANDING 2) CHANNEL DESIGN 3) PRICING"),
        ("1_28", "1) PRICE OF MATERIAL 2) LACK OF PROMOTION IN WARRANTY 3) Brand NAME IS NOT THERE WITH THE PRODCT NAME"),
        ("1_29", "1) PRICE IS HIGHER THAN LOCAL 2) NOT GETTING HIGH MARGIN FROM CUSTOMER"),
        ("1_30", "1) Contractors' find the material costly in comparison with locally available metal.2) If customer doesn't demanding the brand material they refrain to buy our metal.")
    ]

    # Perform the search operation with the search data
    label_result = get_labels(search_data,is_DB=False,pipeline_job_id=0,question_id=11)
    print("Label Result:")
    print(label_result)


def filter_similar_labels(input):
    print("****** filter label ****")

    # Load a pre-trained sentence embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Calculate sentence embeddings
    sentence_embeddings = model.encode(input, convert_to_tensor=True)

    # Calculate cosine similarities between sentence embeddings
    cosine_similarities = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    # Threshold for considering sentences as similar
    similarity_threshold = 0.7

    # Filter out similar sentences
    filtered_sentences = []
    num_sentences = len(input)
    for i in range(num_sentences):
        is_similar = False
        for j in range(num_sentences):
            if i != j and cosine_similarities[i][j] > similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            filtered_sentences.append(input[i])

    # Print the filtered sentences
    print("Before Filter:" + str(input))
    print("After Filter:" + str(filtered_sentences))
    return filtered_sentences



if __name__ == "__main__":
    #main("unit_test_save")
    #main("save_trng")
    main("unit_test_search")


"""
# # Sort the labels based on distance (in ascending order) , reverse = True for distance descending order
        # sorted_labels = sorted(label_score_dict.items(), key=lambda x: x[1], reverse=False)
        # print_debug("sorted result" + str(sorted_labels))

        # # Extract the unique labels based on least distance
        # unique_labels = list({label for label, _ in sorted_labels})
        # print_debug("calling filter_similar_labels")
        # unique_labels_filtered = filter_similar_labels(unique_labels)

        # # Sort the filtered unique labels based on their distances (in ascending order)
        # sorted_filtered_labels = sorted(unique_labels_filtered, key=lambda x: label_score_dict[x], reverse=False)

        # # Get the top 5 filtered labels
        # top5_filtered_labels = sorted_filtered_labels[:5]
        # print_debug("top5 filtered labels " + str(top5_filtered_labels))

 # old code
        # #Sort the labels based on distance (in ascending order) , reverse = True for distance descending order
        # sorted_labels = sorted(label_score_dict.items(), key=lambda x: x[1], reverse=False)
        # print_debug(sorted_labels)
        # # Extract the unique labels based on least distance
        # unique_labels = [label for label, _ in sorted_labels[:top_k]]
        # #unique_labels = [label for label, _ in sorted_labels]
        # print_debug("calling filter_similar_labels")
        # unique_labels_filtered = filter_similar_labels(unique_labels)


        # new code 

        def filter_similar_labels_llm(input):
    print_debug("****** filter label ****")
    print_debug("top 5 labels:" + str(input))
    
    prompt = 
    Given the list of sentences, please find and filter out any sentences that are very similar.
    For Examples
    "Rates on the higher side" and "Price is high" and "cost of product is higher than of local competition" are similar
    "Rates on the higher side" and "brand name not mentioned in product" are NOT similar
    
    model="gpt-3.5-turbo",
    model="davinci"

    # Call the OpenAI API to find similarities
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0,
        max_tokens=100
    )

    # Extract filtered sentences from the response
    filtered_sentences = [sentence for sentence in input if sentence not in response.choices[0].text]

    # Print the filtered sentences
    print_debug("Filtered labels:" + str(filtered_sentences))
    return filtered_sentences

"""