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
from tqdm import tqdm

import sys
import os
import json
import numpy as np
import pickle
from utils import print_debug, exec_sql, pretty_print_json
import utils 
import json
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
ST_SAVE = True  # Set this to True to use the saved ST training model
DISTANCE_FILTER_THRESHOLD=2
DISTANCE_FILTER_THRESHOLD_ENABLED=True
MULTIPLE_ITERATION_MILVUS=True
MULTIPLE_ITERATION_MILVUS_LIFT=1.5 # limit on increase of search count
SEARCH_COUNT_LIMIT = 25 

 
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
print_debug(sys.executable)

import pymysql
model_name="sentence-transformers/all-MiniLM-L6-v2"
collection_name_label = "answers"
top_k=5 

#milvus search results.  i.e for every sentence, how many max search results milvus search should return
# setting this value lower may miss labels if training data has lot of records per label.
#This function initializes the connection to the Milvus server for vector operations.
#It establishes a connection to the Milvus server on the specified host and port.
def init_vector_db():
    
    # Connect to Milvus server
    print_debug("Start connecting to Milvus")
    try:
        connections.connect("default", host="localhost", port="19530")
        print_debug("Successfully connected to Milvus!")
    except Exception as e:
        print_debug("Failed to connect to Milvus:", e)
        sys.exit(1)

#This function retrieves the latest version path for saving a trained model associated with a specific question ID.
#It determines the path where the latest version of the model is stored and returns that path.
def get_latest_version(question_id, next_version=False):
    """
    Get the latest version for saving a trained model.
    Args: question_id (int): The ID of the question.
    Returns: str: The latest version for saving the model.
    """

    # model path , should move to a folder. 
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

#This function retrieves labels for input data using the Sentence Transformer model and vector similarity search.
#It performs a vector search in the Milvus collection to find similar labels for the provided input data.
#The result includes matched labels and their distances from the input data.

# def get_labels new version 
def get_labels(data, is_DB=False, pipeline_job_id=0, question_id=0):
    # Initialize the SentenceTransformer model
    # model = SentenceTransformer(model_name)
    # Check if a saved version of Sentence Transformer exists for the question_id

    # logic: the whole process of getting labels from milvus for raw data has 4 steps
    """ 
       [note: save training takes care of deduping exact labels and lower casing all data in milvus]
        step1: get x labels from milvus basic cosine similarity:  - 
           [x=30 right now, allowing worst case scenario of milvus returning 30 raw sentence of same labels] 
        step2: get rid of labels based on similar labels: 7 of them ["cost high", "price", similar, so only "price" remains] ==> not working
        step3: get only uniqe labels : 4  - working 
        step 4: get top 5 based on ascehding distance - working
    """
    
    saved_version_path = get_latest_version(question_id, next_version=False)
    if saved_version_path is None:
        print_debug("no version is saved")
    else:
        print_debug("saved_version_path " + saved_version_path)

    # If saved version exists, use that model, else use vanilla model
    if saved_version_path and os.listdir(saved_version_path) and ST_SAVE:
        print_debug("Files exisitng in the ST version folder, so loading the saved version")
        model = SentenceTransformer(saved_version_path)
    else:
        print_debug("new question, no saved ST, so starting with vanilla ST")
        model = SentenceTransformer(model_name)

    collection_name = collection_name_label + '_' + str(question_id)
    #print_debug(data)

    # If data is retrieved from the database, extract the raw text from the input
    if is_DB:
        cleaned_data = [(item[0], json.loads(item[1])[0]) for item in data]
        print_debug("cleaned data")
        #print_debug(cleaned_data)
        delete_query = f"DELETE FROM ai.ai_label_vectordb WHERE pipeline_job_id={pipeline_job_id}"
        print_debug("cleaned up ai.ai_label_vectordb at start")
        exec_sql(delete_query)
    else:
        cleaned_data = data

    # Load the collection using the existing schema
    search_collection = Collection(collection_name)
    search_collection.load()

    print_debug(f"answer collection: name: {collection_name} , count {search_collection.num_entities}")

    # Search based on vector similarity
    print_debug("Started searching based on vector similarity")
    search_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",  # Cosine similarity
        "params": {"nprobe": 16},
    }

    # Normalize the embeddings
    embeddings = model.encode([new_answer[1] for new_answer in cleaned_data]).tolist()

    # to be used later  
    progress_sql = f"""UPDATE ai.ai_pipeline_jobs 
                SET job_progress_level = ( 
                SELECT IFNULL(SUM(CASE WHEN job_status='Processed' THEN 1 ELSE 0 END) / COUNT(1), 0)
                FROM ai_pipeline.ai_label_vectordb
                WHERE pipeline_job_id = {pipeline_job_id}
                )
                WHERE pipeline_job_id = {pipeline_job_id}
            """

    # Define a nested function for performing a search
    def perform_search(embeddings, search_count_limit):
        return search_collection.search(embeddings, "embeddings", search_params, limit=search_count_limit,
                                        output_fields=["pk", "answer", "insight_labels"])

    # Perform the search
    search_result = perform_search(embeddings, SEARCH_COUNT_LIMIT)
    print_debug("milvus search result: \n" + str(search_result) + "\n")
    # Apply distance filtering if enabled
    if DISTANCE_FILTER_THRESHOLD_ENABLED:
        filtered_search_result = [[hit for hit in res if hit.distance <= DISTANCE_FILTER_THRESHOLD] for res in search_result]
    else:
        filtered_search_result = search_result

    # Process the search results
    all_new_answers_with_labels = []
    counter = 0 

    print_debug("input data , count = " + str(len(data)))
    for new_answer, result in zip(data, filtered_search_result):
        new_answer_id, new_answer_text = new_answer[0], new_answer[1]

        print_debug("now proessing answer: " + new_answer_text)
        # Extract labels and distances
        labels_and_distances = [(hit.entity.insight_labels, hit.distance) for hit in result]
        #output_details_all = [{"label": label, "distance": distance} for label, distance in labels_and_distances]
        output_details_all = [{"label": hit.entity.insight_labels, "sentence": hit.entity.answer, "distance": hit.distance} for hit in result]

        # Sort and extract unique labels
        sorted_labels = sorted(set(labels_and_distances), key=lambda x: x[1])
        unique_labels = [label for label, _ in sorted_labels]

        # Apply filter_similar_labels function
        unique_labels_filtered = filter_similar_labels(unique_labels)

        # Sort the filtered unique labels based on their distances
        label_distance_dict = {label: distance for label, distance in labels_and_distances}
        sorted_filtered_labels = sorted(unique_labels_filtered, key=lambda x: label_distance_dict.get(x, float('inf')))
        top_k_labels = sorted_filtered_labels[:top_k]

        # Prepare the JSON data for database insertion
        unique_labels_str = json.dumps(top_k_labels)
        output_details_top_json = json.dumps(output_details_all)
        json_labels = json.dumps([{"label": label, "distance": label_distance_dict.get(label)} for label in unique_labels_filtered])

        print_debug("Answer " + new_answer_text)
        print_debug("all labels after removing distance > threshold " + str(output_details_all))
        print_debug("all labels " + str(unique_labels))
        print_debug("filtered out similar labels " + str(unique_labels_str))
        print_debug("output_details_top_json " + str(output_details_top_json))
        
        # If is_DB is True, perform DB operations to save the results
        if is_DB:
            print_debug("Doing DB operation, updating mysql db with labels")
            # Construct the SQL query to save the record
            question_id, answer_id = new_answer_id.split('_')
            
            insert_query = "INSERT INTO ai.ai_label_vectordb (pipeline_job_id, question_id, answer_id, \
                            question_text, answer_text, pipeline_output, pipeline_output_details) \
                            VALUES (%s, %s, %s, %s, %s, %s, %s)" 
            values = (pipeline_job_id, question_id, answer_id, 'N/A', new_answer_text, unique_labels_str, output_details_top_json)
            # delete existing records
            delete_query = f"DELETE FROM ai.ai_label_vectordb WHERE answer_id={answer_id}"
            exec_sql(delete_query)
            exec_sql(insert_query, values)

            
            counter += 1
            if counter % 30 == 0:
                print_debug("******** counter  = " + str(counter) + "**********")
                exec_sql(progress_sql)

        new_answer_with_labels = [new_answer_id, new_answer_text, output_details_all, top_k_labels]
        all_new_answers_with_labels.append(new_answer_with_labels)
    # update crm db
    # update the CRM ai db with the labels
    update_query_crm = f"UPDATE ai_pipeline.ai_label_vectordb a join ai.ai_label_vectordb b \
        ON a.pipeline_job_id=b.pipeline_job_id \
        AND a.answer_id=b.answer_id \
        SET a.pipeline_output= b.pipeline_output, \
            a.pipeline_output_details=b.pipeline_output_details, \
            a.job_status='Processed' \
        WHERE a.pipeline_job_id={pipeline_job_id} \
        AND a.answer_id=b.answer_id \
        AND a.pipeline_job_id=b.pipeline_job_id \
    "
    print_debug(update_query_crm)
    exec_sql(update_query_crm)

    # mark progress to be 100% at end of the process    
    exec_sql(progress_sql)

    # Convert the result to JSON
    json_result = json.dumps(all_new_answers_with_labels, indent=4)
    print_debug("Number of records " + str(len(all_new_answers_with_labels)))
    
    return all_new_answers_with_labels

# we  move the record stats from schduled to Completed skipping inprocess
#This function updates the job status of the processed data to 'Processed' in the label vector database.
#It marks the data associated with a specific pipeline job as processed.
def update_data_for_search(pipeline_job_id):
    sql_ai_user_data = f"UPDATE ai_pipeline.ai_label_vectordb SET job_status='Processed' WHERE pipeline_job_id={pipeline_job_id}"
    exec_sql(sql_ai_user_data)

#This function retrieves data that needs to be processed for label generation from the label vector database.
#It fetches data with a specific job status and returns it as a list of tuples containing answer IDs and text.
def get_data_for_search(pipeline_job_id=0):
    # Retrieve the search data array
    sql = f"SELECT concat(question_id, '_',answer_id) as answer_id,pipeline_input, question_id \
            FROM  ai_pipeline.ai_label_vectordb  \
            WHERE job_status='Scheduled' \
            AND pipeline_job_id={pipeline_job_id}"
    
    status, result = exec_sql(sql)
    
    search_data = []
    for row in result:
        answer_id = row[0]
        answer_text = row[1]
        question_id = row[2]
        search_data.append([answer_id, answer_text])
    
    print_debug("number of raw answers to get labels for: " + str(len(search_data)))
    
    # call validation of training data.  
    # this just displays an warning
    
    return search_data

#This is the main function that orchestrates different operations based on the input parameters.
#It initializes the vector database, retrieves data for processing, generates labels, and updates job statuses.   
def main(operation, pipeline_job_id=0, question_id=0):
    
    print_debug("Search count limit = " + str(SEARCH_COUNT_LIMIT) + " top k=" + str(top_k))

    # Initialize the vector DB
    init_vector_db()
    top_k=5
    if operation == "get_label":
        # Get data for search operation
        search_data = get_data_for_search(pipeline_job_id)
        print_debug(search_data)
        print_debug("\==============")
        # Get labels for search data
        label_result = get_labels(search_data,True,pipeline_job_id, question_id)
        print_debug("Label Result:")
        pretty_print_json(label_result)
        #update_data_for_search(pipeline_job_id)

    elif operation == "unit_test_search":
        unit_test_search()
    else:
        print_debug("Invalid operation. Please specify 'save_trng' or 'get_label' or unit_test_save or unit_test_search.")

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
    
    search_data = [
        # 3 sentences"bagged unfriendy theme", price high theme, store experience bad  and price higher than local theme"
        ("21", "The baggers at the store never smile, always seem rushed. Prices are a bit high compared to other stores nearby. The store layout is confusing, hard to find items. Baggers often seem uninterested in helping. Local markets offer much better deals on several items. Overall, the store experience feels unfriendly and unwelcoming."),
        ("22", "Baggers at the checkout never engage in conversation, very unfriendly. Store prices are significantly higher than I expected. The store's atmosphere is not inviting at all. It's hard to justify the high prices with such poor service. The store doesn't compare well to local competitors. Baggers always seem to be in a bad mood."),
        
        # create 3 feedback records, each has 3 sentences. Each of these records has only 1 theme. 
        ("23", "The baggers at this store are consistently unfriendly. They never offer to help or interact. It feels like they're just going through the motions."),
        ("24", "Prices here are always higher than other places. Can't understand why everything is so expensive. It's not affordable for regular shopping."),
        ("25", "Local stores offer much better prices. This store's pricing is not competitive. You can find better deals elsewhere."),

        # 2 themes from bagged unfriendy theme, store experience not good, price high
        ("26", "The bagger was quite unfriendly during my last visit. Prices seem higher than usual."),
        ("27", "Checkout was not pleasant, baggers were very unfriendly. Store layout is confusing and uninviting."),
        ("28", "Prices are too high for the quality offered. Local stores offer the same items for less.")
    ]
    search_data = [
        ("21", "The baggers are unfriendly. At the store never smile, always seem rushed. The cost of products here is quite steep. Noticed a high price tag on most items, the store experience feels unfriendly and unwelcoming.")
    ]
    search_data = [
        ("21", "The baggers are unfriendly"),
        ("22", "The price is high"),
        ("23", "The baggers are unfriendly.  The baggers are unhelpful too. They are mean."),
        ("24", "The baggers are unfriendly.  The baggers are unhelpful too. They are mean.  The price is high")
    ]
    # Perform the search operation with the search data
    label_results = get_labels(search_data,is_DB=False,pipeline_job_id=0,question_id=99)
    print_debug("Label Result:")
    pretty_print_json(label_results)

# filter out similar labels
def filter_similar_labels(labels, similarity_threshold=0.7):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Remove exact duplicates
    unique_labels = list(set(labels))

    # Vectorize the unique labels
    embeddings = model.encode(unique_labels)

    # Filter out similar labels
    final_labels = []
    excluded_labels = set()  # Track excluded labels

    for i in range(len(embeddings)):
        if unique_labels[i] in excluded_labels:
            continue  # Skip labels already marked as similar

        for j in range(len(embeddings)):
            if i != j and unique_labels[j] not in excluded_labels and cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] > similarity_threshold:
                print(f"Labels too similar: '{unique_labels[i]}' and '{unique_labels[j]}'")
                excluded_labels.add(unique_labels[j])  # Exclude the similar label

        final_labels.append(unique_labels[i])  # Keep the current label
    print_debug("filter labels function: input labels \n" + str(labels))
    print_debug("\nfilter labels function: output labels \n" + str(final_labels))
    return final_labels


if __name__ == "__main__":
    #main("unit_test_save")
    #main("save_trng")
    main("unit_test_search")
