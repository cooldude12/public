"""
usecase : 
Save Training data in milvus
TODO:  load everytime u get collection, remove multiple cursor initiation, do it once in init.  make pk as int
       lower case everything [answer, label] and save it to milvus  
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
import os
import pandas as pd
from torch.utils.data import DataLoader

ST_SAVE = True  # Set this to True to save the trained model

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    IndexType
)
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, InputExample, losses

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
print_debug(sys.executable)

import pymysql
model_name = "distilbert-base-nli-mean-tokens"
model_name="sentence-transformers/all-MiniLM-L6-v2"

def init_vector_db():
    
    # Connect to Milvus server
    print_debug("Start connecting to Milvus")
    try:
        connections.connect("default", host="localhost", port="19530")
        print_debug("Successfully connected to Milvus!")
    except Exception as e:
        print_debug("Failed to connect to Milvus:", e)
        sys.exit(1)
    # Generate embeddings using TF-IDF vectorizer

def check_collection_exist(collection_name_passed):

    # drop a collection 
    DROP_COLLECTION=True
    if DROP_COLLECTION:
        utility.drop_collection(collection_name_passed)
        print_debug("collection Dropped")

    # Check if the collection exists in Milvus
    has_collection = utility.has_collection(collection_name_passed)
    if not has_collection:
        print_debug(f"Collection '{collection_name_passed}' does not exist, creating collection...")

        model = SentenceTransformer(model_name)
        # Create the collection
        num_entities, dim_value = 3000, 384
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, description="ID of the answer"),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, description="The answer text", max_length=200),
            FieldSchema(name="insight_labels", dtype=DataType.VARCHAR, description="Insight labels for the answer", max_length=100),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim_value, description="Answer embeddings"),

        ]
        schema = CollectionSchema(fields, description="Collection of answers")
        save_collection = Collection(name=collection_name_passed, schema=schema,consistency_level="Strong")
        print_debug("collection has been created")

        # Create index
        index_name = "embeddings"
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        # Drop the index if it exists
        index_name = "embeddings"
        #if save_collection.has_index(index_name):
        #save_collection.drop_index(index_name)
        #print_debug("Dropped index " + index_name)
        save_collection.create_index(index_name, index)

        print_debug(f"Collection '{collection_name_passed}' created successfully with index '{index_name}'")
    else:
        print_debug(f"Collection '{collection_name_passed}' already exists.")

# validate the training data for the question.  
# we should see increasing count of number of labels for job_ids
# since labels are additive. 
def validate_training_data(question_id):
    if not question_id or question_id == 0:
        print_debug("question id is null, validation check skipped")
        return
    
    sql_check = f"""
        SELECT 
        a.pipeline_job_id, 
        COUNT(1) as cnt,
        CASE 
            WHEN LAG(COUNT(1)) OVER (ORDER BY a.pipeline_job_id) > COUNT(1) THEN 'warning' 
            ELSE 'ok' 
        END AS message
        FROM ai_pipeline.ai_save_training_data a
        WHERE 
            a.question_id = {question_id}
        GROUP BY a.pipeline_job_id
        ORDER BY a.pipeline_job_id desc
    """
    status, result = exec_sql(sql_check)
    print_debug("training data for this question id over the pas jobs:")
    print_debug(result)

    if status and result:
        first_record = result[0]
        message = first_record[2]

        if message == 'warning':
            print_debug("Training data looks incorrect. Please check if all answers \
                        for question uploaded. Sleeping for 5 seconds...")
            time.sleep(5)
        else:
            print_debug("validation of training data check passed")

def check_existing_data(collection_name, id_list):
    collection = Collection(collection_name)
    collection.load()

    # Fetch existing entities from the collection
    # this does not work for unit test wkflow
    existing_entities = collection.query(expr="pk in " + str(id_list), output_fields=["pk"])
    print_debug("Number of existing entities to match: " + str(len(existing_entities))) 
    print_debug("Existing entities: " + str(existing_entities))

    # Check if any existing entities match the input IDs
    if len(existing_entities) == 0:
        print_debug("No existing entities matched the input IDs. Skipping deletion.")
        return

    # Extract the primary keys from the existing entities
    existing_primary_keys = [entity['pk'] for entity in existing_entities]

    # Create a list of primary keys to be deleted
    primary_keys_to_delete = [f"'{pk}'" for pk in existing_primary_keys]

    # Construct the expression for deletion
    expr_str = f"pk in [{', '.join(primary_keys_to_delete)}]"

    # Delete existing entities
    num_deleted = collection.delete(expr_str)
    print_debug("Number of entities deleted: " + str(num_deleted))
    return

def save_training_data(training_data, question_id, operation="save_training_data"):
    # Initialize the SentenceTransformer model
    print_debug("operation = " + operation)
    model = SentenceTransformer(model_name)
    collection_name = "answers"
    collection_name = collection_name + '_' + str(question_id)
    print_debug("Collection name = " + collection_name)

    # Drop the collection if it exists, do it only for unit_test
    if operation == "unit_test":
        print_debug("dropping Collection , only  happens during unit test")
        utility.drop_collection(collection_name)

    # Check if the collection exists and create it if necessary
    check_collection_exist(collection_name)

    # Create a list of entities from the training data
    entities = [
        [item[0] for item in training_data],  # primary key field
        [item[1] for item in training_data],  # answer field
        [item[2] if item[2] else None for item in training_data],  # insight_labels field, taking the first label if exists
        model.encode([item[1] for item in training_data]).tolist(),  # embeddings field
    ]

    # delete existing entries 
    print_debug("now checking for if ids already existst, then delete them before inserting")
    input_primary_keys = [item[0] for item in training_data]
    print_debug(input_primary_keys)
    if operation != "unit_test":
        check_existing_data(collection_name, input_primary_keys)
    else:
        print_debug("unit test, skipping the check existing data, since collection is reset everytime")

    # Connect to the collection
    save_collection = Collection(name=collection_name)
    
    # Save all input entities
    print_debug("now going to save the data")
    #print_debug(entities)

    #save_collection.insert(entities, batch_size=1000)
    total_entities = len(entities[0])
    print_debug("total entities to save" + str(total_entities))
    batch_size=256

    for i in range(0, total_entities, batch_size):
        print_debug("i = " + str(i))
        batch_entities = [
            entity_list[i:i + batch_size] for entity_list in entities
        ]
        save_collection.insert(batch_entities)
        print_debug("saved milvus , batch num " + str(i))
    # Commit changes
    save_collection.flush()

    # Print the number of entities
    print_debug(f"Post insert: Number of entities in Milvus: {save_collection.num_entities}")

    # Print some sample results
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    save_collection.load()
    results = save_collection.search(np.random.random((3, model.get_sentence_embedding_dimension())), "embeddings",
                                     search_params, limit=3, output_fields=["answer", "insight_labels"])
    print_debug("Getting some sample data post creation to do some spot checking")
    print_debug(results)
    return True

def get_data_for_save(pipeline_job_id=0):
    # SQL query to fetch data from MySQL
    # dedup training data: 
    print_debug("getting data for save ")
    sql_delete = f"DELETE FROM ai_pipeline.ai_save_training_data_dedup  WHERE pipeline_job_id={pipeline_job_id} ;"
    sql_dedup = f""" 
        INSERT INTO ai_pipeline.ai_save_training_data_dedup (pipeline_job_id,job_status, vertical_id, 
                sub_vertical_id, question_id, pipeline_input, label) 
        SELECT  {pipeline_job_id}, 'Scheduled', vertical_id, sub_vertical_id, question_id, pipeline_input, label 
        FROM ( 
            SELECT vertical_id, sub_vertical_id, question_id, lower(pipeline_input) as pipeline_input, 
                lower(label) as label, MIN(id) AS id 
            FROM ai_pipeline.ai_save_training_data 
            WHERE pipeline_job_id = {pipeline_job_id} 
            AND job_status = 'Scheduled'
            GROUP BY vertical_id, sub_vertical_id, pipeline_input, label 
        ) a; 
    """
    
    print_debug("\n --- Now saving in training master ---- ")
    # save also in a master training, do an upsert
    sql_training_master = f"""
        INSERT INTO ai_pipeline.ai_training_master (vertical_id, sub_vertical_id, 
        question, answer, training_label)
        SELECT vertical_id, sub_vertical_id, b.question, a.pipeline_input, a.label 
        FROM ai_pipeline.ai_save_training_data_dedup a 
        JOIN triestai.questions b ON a.question_id=b.question_id
        WHERE a.pipeline_job_id = {pipeline_job_id}
        ON DUPLICATE KEY UPDATE training_label = VALUES(training_label)
    """
    exec_sql(sql_training_master)
    sql_question = f"SELECT question_text FROM ai_pipeline.ai_save_training_data \
                        WHERE pipeline_job_id = {pipeline_job_id}"
    
    status, result = exec_sql(sql_question)
    question_id_training_data=""
    for row in result:
            question_training_data = row[0]
    print_debug("question txt for trainign data = " + str(question_training_data))
    # 1_1 is hard coded for vertical and sub vertical
    # sql_get_data = f"SELECT CAST(id AS CHAR) as id, rtrim(ltrim(pipeline_input)) as pipeline_input, \
    #         rtrim(ltrim(label)) as answer_label, pipeline_job_id \
    #         FROM ai_pipeline.ai_save_training_data_dedup \
    #         WHERE job_status = 'Scheduled'  \
    #             AND ltrim(rtrim(label)) IS NOT NULL and label != '' \
    #             AND pipeline_job_id={pipeline_job_id} \
    #            "   
    # Escape single quotes in the variable
    question_training_data_escaped = question_training_data.replace("'", "''")
 
    sql_get_data = f"""
            SELECT id, rtrim(ltrim(answer)) as pipeline_input, 
            rtrim(ltrim(training_label)) as answer_label, {pipeline_job_id} 
            FROM ai_pipeline.ai_training_master 
            WHERE ltrim(rtrim(training_label)) IS NOT NULL and training_label != '' 
                            AND question=LEFT('{question_training_data_escaped}', 300)
    """   
    print_debug(sql_get_data)
    try:
        # Execute the SQL query and fetch results
        status, result = exec_sql(sql_delete)
        status, result = exec_sql(sql_dedup)
        status, result = exec_sql(sql_training_master)
        status, result = exec_sql(sql_get_data)
        # Initialize the array for save data
        num_records = len(result)
        print_debug("sql executed, " + str(status) + " number of records :" + str(num_records))
        save_data = []

        # Iterate over the fetched records
        for row in result:
            id = row[0]
            answer = row[1]
            insight_label = str(row[2]) 
            # Append the data to the save_data array
            save_data.append([id, answer, insight_label])
        print_debug("number of recrods for training data = " + str(len(save_data)))
        print_debug("**** Data to save *******")
        print_debug(save_data)
            
        return save_data
    except Exception as e:
        print_debug("Error occurred while fetching data from MySQL:" + str(e))
        sys.exit(1)

# update the records at end of save
# we  move the record stats from schduled to Completed skipping inprocess
def update_data_for_save(pipeline_job_id):
    print_debug("updating training data job status to processed")
    sql = f"UPDATE ai_pipeline.ai_save_training_data SET job_status='Processed' \
        WHERE pipeline_job_id={pipeline_job_id}"
    exec_sql(sql)

# get the latest version of the folder
def get_latest_version(question_id, next_version=False):
    #base_path = os.path.abspath('../../st_models')
    base_path = os.environ.get('ST_MODELS_PATH', '/home/ec2-user/code/st_models')
    question_path = os.path.join(base_path, str(question_id))

    print_debug("get_latest_version: question_id:" + str(question_id) + " next version:" + str(next_version) + \
                " question path:" + question_path + " base path:" + base_path)

    if not os.path.exists(question_path):
        os.makedirs(question_path)
        if next_version:
            version_path = os.path.join(question_path, 'v1')
            print_debug("version path " + version_path)
            os.makedirs(version_path)
            print_debug("folder created: " + version_path)

        else:
            return None

    else:
        version_folders = [folder for folder in os.listdir(question_path) if folder.startswith('v')]
        if not version_folders:
            new_version = 1
        else:
            latest_version = max([int(folder[1:]) for folder in version_folders])
            if next_version:
                new_version = latest_version + 1
            else:
                new_version = latest_version
        version_path = os.path.join(question_path, f'v{new_version}')
        print_debug("version path:" + str(version_path))
        if next_version:
            os.makedirs(version_path)
            print_debug("folder created: " + version_path)

    return version_path


def save_st_model(training_data, question_id):
    """
    Train a SentenceTransformer model and save it to disk.
    Args: question_id (int): The ID of the question.
    """

    if not ST_SAVE:
        print_debug("ST_FLAG is set to False. Skipping model saving.")
        return
    
    # since we are loading the cumulative/all answers for the question_id 
    # every time, no need to retrive the saved model, as training data being 
    # loaded has all the  past data.  
    # v1: 100 cold start label
    # v2: 10 deleted, 20 new, 30 updated, so v2 count: 100 - 10 + 20 + 30 = 140 
    # so while training v2, we retrieve v1 that has 100 labels and do 140 on top of it
    # though 90 of the 140 is already in v1 
    
    saved_version_path = get_latest_version(question_id, next_version=False)
    if saved_version_path is None:
        print_debug("no version is saved, trying vanilla ST")
        model = SentenceTransformer(model_name)
    else:
        print_debug("using save ST from saved_version_path " + saved_version_path)
        model = SentenceTransformer(saved_version_path)

    # now find the location of saving the next version
    model_path = get_latest_version(question_id, next_version=True)
    print_debug("next ST version path " + model_path)

    train_examples = []
    print_debug("lenght of training data = " + str(len(training_data)))
    for example in training_data:
        id, feedback_text, label = example
        train_examples.append(InputExample(texts=[feedback_text, label], label=0.9))

    print_debug("ST Training: step1: dataloader starting ")        
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    print_debug("ST Training: step2: train_loss starting ") 
    train_loss = losses.CosineSimilarityLoss(model)
    print_debug("ST Training: step3: model.fit starting ")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    print_debug("ST Training: step4: model.save starting ")
    model.save(model_path)
    print_debug("ST Training: done")


def main(operation, pipeline_job_id=0, question_id=0):
    # Initialize the vector DB
    
    if operation == "save_training_data":
        init_vector_db()

        # Get data for save operation
        # Save new data with labels
        validate_training_data(question_id)

        training_data = get_data_for_save(pipeline_job_id)
        save_result = save_training_data(training_data, question_id, operation)
        
        # for performance reason divide the training data in batches 
        batch_size=50
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]
            print_debug("calling save_st repteadly. currnet index is at record#: " + str(i))
            save_st_model(batch, question_id)
            print_debug("count of saved data in save model in this run: " + str(batch_size))

        print_debug("all save ST done with batch size " + str(batch_size))
        update_data_for_save(pipeline_job_id)

    elif operation == "unit_test_save":
        unit_test_save()
    else:
        print_debug("Invalid operation. Please specify 'save_training' or 'get_label' or unit_test_save or unit_test_search.")

# unit test
def unit_test_save():
    # Retrieve the save data array
    question_id=99

    save_data = [
        ("1_1_1", "The pizza was absolutely delicious, the toppings were so fresh and plentiful!",  "delicious pizza"),
        ("1_1_2", "Loved the pizza! The crust was cooked to perfection.", "perfect crust"),
        ("1_1_3", "The cheese was just so creamy and the toppings were a treat to my taste buds!", "tasty toppings"),
        ("1_1_4", "Pizza was alright, the crust could have been a bit more crisp.", "crust not crisp"),
        ("1_1_5", "The cheese on the pizza was heavenly, but I found the toppings a bit lacking.", "bad toppings"),
        ("1_1_6", "The pizza was just average, nothing exceptional.", "average pizza"),
        ("1_1_7", "I loved the toppings but the crust was not up to the mark.", "tasty toppings"),
        ("1_1_8", "The pizza was okay. I've had better.", "average pizza"),
        ("1_1_9", "Did not enjoy the pizza, the toppings didn't seem fresh.", "bad toppings"),
        ("1_1_10", "The crust was too hard and the cheese was not melted properly.",  "unmelted cheese"),
        ("1_1_11", "The pizza was too greasy for my taste.", "pizza greasy"),
        ("1_1_12", "The cheese was stale and the toppings were scarce.",  "scarce toppings"),
        ("1_1_13", "Did not enjoy the pizza. The crust was undercooked.", "undercooked crust"),
        ("1_1_14", "The pizza tasted stale, was not satisfied.", "stale pizza"),
        ("1_1_15", "The toppings were too few and the cheese didn't taste fresh.", "few toppings"),
        ("1_1_16", "The crust was perfect but the toppings were not fresh.", "not fresh toppings"),
        ("1_1_17", "The pizza was just about alright, not what I expected.", "average pizza"),
        ("1_1_18", "Loved the amount of cheese on the pizza, but the crust was not cooked properly.",  "uncooked crust"),
        ("1_1_19", "The crust was too thick and the toppings were not evenly spread.", "thick crust"),
        ("1_1_20", "I did not like the pizza, the cheese was not good.", "bad cheese")
    ]
    save_data = [
       ("1_1_1", "The pizza was absolutely delicious, the toppings were so fresh and plentiful!",  "delicious pizza"),
       ("1_1_21", "best pizza i ever had with yummy toppings", "tasty toppings"),
        ("1_1_22", "Pizza was just bad in taste and expesive.", "price expensive")
    ]
    save_data1 = [
       ("1_1_1", "The pizza was absolutely delicious, the toppings were so fresh and plentiful!",  "delicious pizza"),
       ("1_1_21", "best pizza i ever had with yummy toppings", "tasty toppings"),
       ("1_1_22", "Pizza was just bad in taste and expesive.", "price expensive")
    ]
    # bldg material
    save_data = [
        ("1_1_1", "Cost is somewhat high compare to some local metals", "Cost of product is higher than of local competition"),
        ("1_1_2", "No Knurling on material", "Product doesn't have Knurling"),
        ("1_1_3", "Price is high compare to some other local", "Cost of product is higher than of local competition"),
        ("1_1_4", "No brand on metal", "Brand name is not mentioned on product"),
        ("1_1_5", "No Warranty on product", "Product doesn't come with warranty"),
        ("1_1_6", "COMPANY BRANDING", "Brand name is not mentioned on product"),
        ("1_1_7", "CHANNEL DESIGN", "Product design"),
        ("1_1_8", "PRICE IS HIGHER THAN LOCAL", "Cost of product is higher than of local competition"),
        ("1_1_9", "NOT GETTING HIGH RATES FROM CUSTOMER", "Cost of product is higher than of local competition"),
        ("1_1_10", "PRICE DIFFERENCE", "Rates on the higher side"),
        ("1_1_11", "NO Brand NAME", "Brand name is not mentioned on product"),
        ("1_1_12", "No Brand Variety and sizes", "Product sizes that required is not available"),
        ("1_1_13", "No Brand Name written on section.", "Brand name is not mentioned on product"),
        ("1_1_14", "expensive then local metal.", "Cost of product is higher than of local competition"),
        ("1_1_15", "PRICE OF MATERIAL", "Rates on the higher side"),
        ("1_1_16", "LACK OF PROMOTION IN WARRANTY", "Product doesn't come with warranty"),
        ("1_1_17", "Brand NAME IS NOT THERE WITH THE PRODCT NAME", "Brand name is not mentioned on product"),
        ("1_1_18", "Light weight sizes not compare to local metal", "Product sizes is not available"),
        ("1_1_19", "Rates,", "Rates on the higher side"),
        ("1_1_20", "Credit not available", "Credit terms"),
        ("1_1_21", "Price", "Rates on the higher side"),
        ("1_1_22", "No brand name on metals", "Brand name is not mentioned on product"),
        ("1_1_23", "price higher as compared to local", "Cost of product is higher than of local competition"),
        ("1_1_24", "no branding", "Brand name is not mentioned on product"),
        ("1_1_25", "warranty is required", "Product doesn't come with warranty"),
        ("1_1_26", "*cost is high.", "Rates on the higher side"),
        ("1_1_27", "Brand name missing", "Brand name is not mentioned on product"),
        ("1_1_28", "No brand name", "Brand name is not mentioned on product"),
        ("1_1_29", "High price when compares to some other local metals,", "Cost of product is higher than of local competition"),
        ("1_1_30", "Credit terms", "Credit terms"),
        ("1_1_31", "PRICE OF PRODUCT.", "Rates on the higher side"),
        ("1_1_32", "AVAILABILITY OF MATERIAL AT SUB DEALER COUNTERS.", "Product sizes is not available"),
        ("1_1_33", "MANY CONTRACTORS DID NOT UNDERSTAND THE DESIGN OF THE PRODUCT.", "Product design"),
        ("1_1_34", "Contractors' find the material costly in comparison with locally available metal.", "Cost of product is higher than of local competition"),
        ("1_1_35", "If customer doesn't demanding the brand material they refrain to buy our metal.", "Lack of customer demand"),
        ("1_1_36", "No name on metals", "Brand name is not mentioned on product"),
        ("1_1_37", "pricing", "Rates on the higher side"),
        ("1_1_38", "strength of the ceiling section", "Product Quality not up to par"),
        ("1_1_39", "branding not there", "Brand name is not mentioned on product"),
        ("1_1_40", "BRANDING ON THE SECTION", "Brand name is not mentioned on product"),
        ("1_1_41", "DEMAND FROM THE CUSTOMER", "Lack of customer demand"),
        ("1_1_42", "brand perception of weight driven metal", "Brand name is not mentioned on product"),
        ("1_1_43", "Brand is not getting requested", "Brand name is not mentioned on product"),
        ("1_1_44", "local section is available in better thickness only for ceiling", "Product sizes is not available"),
        ("1_1_45", "its price", "Rates on the higher side"),
        ("1_1_46", "its overall design looks very low in same category if I represent local .45 material and that is also cheaper than expert", "Product Quality not up to par"),
        ("1_1_47", "Lack of awareness", "Lack of customer demand"),
        ("1_1_48", "Contractors/Customer only want Knurled Item", "Product doesn't have Knurling"),
        ("1_1_49", "Branding (Missing Name on Brand)", "Brand name is not mentioned on product"),
        ("1_1_50", "Weight / Thickness issue", "Product Quality not up to par"),
        ("1_1_52", "PRICE HIGHER THEN LOCAL", "Cost of product is higher than of local competition"),
        ("1_1_53", "QUALITY LOWER THEN LOCAL THICKNESS", "Product Quality not up to par"),
        ("1_1_54", "Price", "Rates on the higher side"),
        ("1_1_55", "Angle and Perimeter are very lightweight", "Product Quality not up to par"),
        ("1_1_56", "Prices are little bit higher than other", "Cost of product is higher than of local competition"),
        ("1_1_57", "Brand not mentioned", "Brand name is not mentioned on product"),
        ("1_1_58", "Quality is lacking", "Product Quality not up to par"),
        ("1_1_59", "Price", "Rates on the higher side"),
        ("1_1_60", "Product quality", "Product Quality not up to par"),
        ("1_1_61", "No Knurling on product", "Product doesn't have Knurling"),
        ("1_1_62", "Quality not as good as competitive brands", "Product Quality not up to par"),
        ("1_1_63", "Price", "Rates on the higher side"),
        ("1_1_64", "No mention of BRAND NAME SHOULD BE THERE ON PRODUCT", "Brand name is not mentioned on product"),
        ("1_1_65", "Quality is not that good", "Product Quality not up to par"),
        ("1_1_66", "Price of BRAND is very High", "Rates on the higher side"),
        ("1_1_67", "Its not a Knurling Channel", "Product doesn't have Knurling"),
        ("1_1_68", "Local the Manufacturers are selling sizes which not available in brand", "Product sizes is not available"),
        ("1_1_69", "Rate Compared to Local Metals for the same Thickness", "Cost of product is higher than of local competition"),
        ("1_1_70", "Quality of Product", "Product Quality not up to par"),
        ("1_1_71", "Price", "Rates on the higher side"),
        ("1_1_72", "Quality of material not up to the mark.", "Product Quality not up to par"),
        ("1_1_73", "No availability of required sizes.", "Product sizes is not available"),
        ("1_1_74", "Prices Higher", "Cost of product is higher than of local competition"),
        ("1_1_75", "No branding", "Brand name is not mentioned on product"),
        ("1_1_76", "No brand", "Brand name is not mentioned on product"),
        ("1_1_77", "Quality", "Product Quality not up to par"),
        ("1_1_78", "Product not available when needed.", "Product sizes is not available"),
        ("1_1_79", "No Varieties and sizes available", "Product sizes is not available"),
        ("1_1_80", "Price is high as compared to local market", "Cost of product is higher than of local competition"),
        ("1_1_81", "No proper marketing", "Product design"),
        ("1_1_82", "Price is high compared to local", "Cost of product is higher than of local competition"),
        ("1_1_83", "No branding by the company", "Brand name is not mentioned on product"),
        ("1_1_84", "Price is higher than the local market", "Cost of product is higher than of local competition"),
        ("1_1_85", "No brand value", "Brand name is not mentioned on product"),
        ("1_1_86", "Price is too high", "Rates on the higher side"),
        ("1_1_87", "No demand due to local availability of the same product", "Lack of customer demand"),
        ("1_1_88", "Price is very high", "Rates on the higher side"),
        ("1_1_89", "No variety in sizes", "Product sizes is not available"),
        ("1_1_90", "Product branding", "Product design"),
        ("1_1_91", "Price", "Rates on the higher side"),
        ("1_1_92", "Quality difference", "Product Quality not up to par"),
        ("1_1_93", "Product design", "Product design"),
        ("1_1_94", "Quality", "Product Quality not up to par"),
        ("1_1_95", "Product is not as per requirement", "Product sizes is not available"),
        ("1_1_96", "Quality of product", "Product Quality not up to par"),
        ("1_1_97", "Product rates", "Rates on the higher side"),
        ("1_1_98", "Quality", "Product Quality not up to par"),
        ("1_1_99", "Product variety", "Product sizes is not available"),
        ("1_1_100", "Rates are high", "Rates on the higher side"),
        ("1_1_101", "Variety of sizes not available", "Product sizes is not available"),
        ("1_1_102", "Rates", "Rates on the higher side"),
        ("1_1_103", "Variety of sizes not available", "Product sizes is not available"),
        ("1_1_104", "Rates are high", "Rates on the higher side"),
        ("1_1_105", "Variety of sizes not available", "Product sizes is not available"),
        ("1_1_106", "Some contractors demand complete branded items", "Lack of customer demand"),
        ("1_1_107", "Product quality", "Product Quality not up to par"),
        ("1_1_108", "The product is not up to the mark as the local products.", "Product Quality not up to par")
    ]
    # deca improvement comments
    save_data = [
        ("1_1_1","Produce selection needed more variety and competitive pricing. ","Need more variety of Produce"),
        ("1_1_2","We need more frequently if there were more sales.","Need more sale / coupons"),
        ("1_1_3","Many items were not available.  Boxed items, such as cookies, are much cheaper at civilian supermarkets.  ","Products not on shelf"),
        ("1_1_4","Probably the availabillity of more fruits that are already sliced would be nice, like: pineapple!","Need more variety of Produce"),
        ("1_1_5","(1) I regularly use my rewards card for its digital coupons.  Today, one of my $1 digital coupons (48 poise liners) was not honored at checkout.  This is not an infrequent occurrence, and it is disappointing when it happens!  ","Coupons issues"),
        ("1_1_6"," Another concern I have is what I'm told re: what I saved by shopping at the commissary.  One helpful improvement could be added to the receipt: Cost of the item at the commissary, plus what the cost of the item that it is being compared to on the economy.  The technology for doing this is obviously available and would give more credibility to your marketing campaign.","Show price saving on receipt"),
        ("1_1_7","Beautiful store , I always enjoy my shopping experience , great customer service , great management team .","Positive comment"),
        ("1_1_8","Continue the positive and pleasant atmosphere of this commissary","Positive comment"),
        ("1_1_9","iteem availability is often inconsistent","Products not on shelf"),
        ("1_1_10","i used to purchase knorr brand german salad mix packets like dill krauter you no longer carry knorr vegetable and beef  bouillon  also dialbody wash marula oil and ham slices made by smithfield.","Bring back discontinued products"),
        ("1_1_11","Having self checkout only on Mondays causes delays in line and makes it difficult when you have a lot of groceries. Having at least 1-2 checkout lines open during Mondays would help.","Increase check out lines and number cashiers"),
        ("1_1_12","I appreciate the employees making me feel as if I'm not just another customer by being attentive to my questions","Increase attentiveness towards customers"),
        ("1_1_13","Quality products that I habitually buy disappear never to return again.  These are everyday items like coffee maker descaler, jimmy dean sausage crumbles, certain salad dressings, unfrozen cooked chicken for salads, certain coffee brands.  More consistancy in products offered would be welcomed.","Bring back discontinued products"),
        ("1_1_14","The shelves are never fully stocked. ","Products not on shelf"),
        ("1_1_15","Hard to find what I need. ","Products not on shelf"),
        ("1_1_16"," Never any music playing. Feels like a cold environment. Not warm and inviting.","Poor store shopping experience"),
        ("1_1_17","I used to shop at the commissary on a weekly basis. Slowly, items that I usually bought were not available anymore. This happens now every time I go shopping. Are you planning on shutting this commissary down and are slowly pushing us to shop elsewhere?","Bring back discontinued products"),
        ("1_1_18","The commissary closes 1 hour earlier, this really impacts my decision whether I have enough time to get to the commissary and do my shopping. Then I say the heck with it and I go to my local store.If the commissary starts stocking things up again,","Bring back discontinued products"),
        ("1_1_19"," Please, I want my commissary back, please tell the manager, it's ok to be nice.","Associates need to be friendly towards customers"),
        ("1_1_20","More bread options, I miss the variety of bagels.","Need more bakery products"),
        ("1_1_21","Need more Fresh produce to come in! ","Need more variety of Produce"),
        ("1_1_22","Prices should be lower than off base!","Lower prices"),
        ("1_1_23","The lady working the checkout station is a rude and unhelpful associate. Where did the commissary find her? Very unpleasant!!","Associates need to be friendly towards customers"),
        ("1_1_24","friendliness of employees. employees are often rude or have attitudes. i don't know if it's the work environment or them personally but almost everytime i come to the commissary or BX the staff is rude, unfriendly, and unhappy. very unwelcoming.","Associates need to be friendly towards customers"),
        ("1_1_25","Why do I have to show my ID every time I use the self checkout at JBER?  It's almost to the point of harassment.  My hands are full and I'm asked/told to drop everything, dig in my wallet, pull out my ID to show it to someone, then have to scan it.  Why?  JBER has to have the worst self-check out policy of any Commissary I've been to and I've been to many.","Decrease number times Check ID "),
        ("1_1_26","There is no maximum number of items allowed for self-checkout so the line becomes ridiculously long and isn't as quick as intended. Shoppers bring full carts, sometimes 80+ items, and take 30 minutes to checkout. The attendant should direct them to the manned cashier lines. ","Separate checkout for large number of items"),
        ("1_1_27","Additionally, retirees and other non-active duty shoppers consistently buy the entire stock of items before active members and their dependents have a chance to purchase anything. Anchorage has a high cost of living, so it is unfair that those who CHOSE to live here take advantage of the slightly cheaper prices while active duty members struggle or go without certain items because of it.","Active duty should preference"),
        ("1_1_28","1-PLEASE fix the price tags on the shelves. Some of them on higher shelves point to the ceiling and cannot be read. And some on the lower shelves point towards the ground and cannot be read.","Fix price tag on shelf"),
        ("1_1_29","2-some items on higher shelves and lower shelves are pushed all the way to the back and can't be reached.","Can't reach items on shelf"),
        ("1_1_30","3-PLEASE fix the seafood/bread freezer. It is STILL reading just below freezing, borderline dangerous temperature for seafood. When you open one of the doors you can SMELL the thaw out in the freezer. PLEASE FIX!","Freezer not working properly"),
        ("1_1_31","Consistent on product availability. ","Products not on shelf"),
        ("1_1_32","More produce.","Need more variety of Produce"),
        ("1_1_33","Have more cashiers come to work cos some of us dont want to stand that long waiting. I waited in line longer than it took to shop!","Increase check out lines and number cashiers"),
        ("1_1_34","More selection and variety","Need more variety of Products options"),
        ("1_1_35","Variety of vegetables would be nice.","Need more variety of Produce"),
        ("1_1_36","A wider variety of products.","Need more variety of Products options"),
        ("1_1_37","Better selection of Goya and other international products.","Need more variety of Products options"),
        ("1_1_38","Some items sell out faster than others on a regular basis and are often missing on the shelf. They should be ordered in greater amounts.","Products not on shelf"),
        ("1_1_39","This commissary routinely runs out of stock. There are always empty shelves.","Products not on shelf"),
        ("1_1_40","Many items at or near expiration date","Expiring Items "),
        ("1_1_41","Just want regular grocery, not high end products that are expensive. We do purchase store brand when possible and if they are not more expensive than national brands.","Lower prices"),
        ("1_1_42","notice that several of the items I purchase are no longer available...","Bring back discontinued products"),
        ("1_1_43","The cashier overcharged us several times by inputting the wrong produce, failing to recognize a markdown, saying a coupon applied when it hadn't.","Overcharged on products"),
        ("1_1_44","At 1000 on a payday Saturday morning and there is only ONE cashier line open with 6-7 customers waiting in line for checkout.  This is a recurring problem and quite frankly there seems to be no concern with this from the commissary management.  Obviously management does not see the very unhappy patrons waiting in line.  Still more items are apparently no longer being carried, continuing the trend to less choice and lesser quality items on the shelf in some areas.","Increase check out lines and number cashiers"),
        ("1_1_45","nothing obvious other then the shelves were better stocked with products","Positive comment"),
        ("1_1_46","I am becoming increasingly disappointed in empty shelves for products I wan to buy!  This time there wasn't any Morton Lite Salt or  Hebrew National Jumbo hotdogs.  My last shopping trip there were these two items, plus two others I couldn't get.  You have signs up about supply chain problems, but I can go to Safeway or Walmart and find those items on their shelves.  Sometimes it could be a month or more before and item returns to your shelves.  Then, on the next trip, they are gone again.","Products not on shelf"),
        ("1_1_47","Commissary no longer carries Ralston cereals","Bring back discontinued products"),
        ("1_1_48","I love shopping at the Bangor Commissary over the other one in the area. The selection is better, people are friendlier and there's much better parking!","Positive comment"),
        ("1_1_49","Deli staff gave me incorrect product.  Asked specifically for one brand and was given another, the brand received significantly more expensive than the requested and in-stock brand.  ","Order not fullfilled correctly"),
        ("1_1_50","  Insufficient cashiers and baggers on hand.","Increase check out lines and number cashiers"),
        ("1_1_51","This commissary has continuous out of stock problems.  Tried to buy a roast chicken... normally carried at every grocery store.  Commissary had none and the deli rotisserie was not even running.  Had to change entire dinner plans based on that.   ","Need more variety of Products options"),
        ("1_1_52"," I got lucky and could use the express checkout..... 5 or 6 people waiting in line for an open checkout register.  This commissary continues to short staff the checkout registers and make patrons wait excessive times for an open register.","Increase check out lines and number cashiers"),
        ("1_1_53","The deli workers are constantly busy and deal with a lot of customers ","Less staff in Deli section"),
        ("1_1_54","ENSURE PRODUCTS AVAILABLE ARE NOT ONLY OF QUALITY BUT TO ENSURE PRODUCTS PAST EXPIRATION DATES OR THOSE THAT DO NOT MEET THE STANDARD ARE NOT ON SHELVES. ALSO, MAINTAIN THE SHELFS OF THOSE PRODUCTS.","Expiring Items "),
        ("1_1_55","For the second time the deli staff incorrectly filled out my order, giving me a very expensive brand of cold cut VICE the specific (less expensive) one I asked for.  Your deli staff must be hard of hearing or just plain inattentive.","Order not fullfilled correctly"),
        ("1_1_56","In this commissary the fruit and produce section has been cut back to bare minimum.  And the meat department is a hit or miss. It would be nice to see it back to what it was acouple of years ago. At that point I wouldn't need to go to another store to complete my shopping.","Need more variety of Produce"),
        ("1_1_57","Sometimes doing nothing is doing something.  Make sure you are taking care of your people.","Associates need to be friendly towards customers"),
        ("1_1_58","Consistent lack of available cashiers is unacceptable, in this instance there was only a single cashier at 1030, resulting in an almost 20 minute wait.  Manager seems to be nonexistent.  This store continues to decline in quality/reliability.","Increase check out lines and number cashiers"),
        ("1_1_59","Keep up the good work !!","Positive comment"),
        ("1_1_60","I would love to see more Tillamook products.","Need more variety of Produce"),
        ("1_1_61","Keep up the great work","Positive comment"),
        ("1_1_62","every time seems  prices is going up.","Lower prices"),
        ("1_1_63","Baggers are a great assets","Positive comment"),
        ("1_1_64","Cashiers and Baggers are why I shop at this commissary store.  They are the best.  Making the shopping experience pleasant..","Positive comment"),
        ("1_1_65","Cashier and Baggers Great","Positive comment"),
        ("1_1_66","Baggers are a real bonus.","Positive comment"),
        ("1_1_67","Please bring back vegan sausages and pickled okra","Bring back discontinued products"),
        ("1_1_68","Some products expiration date is only a few days after buying product.","Expiring Items "),
        ("1_1_69","More variety for gluten free products","Need more variety of Produce"),
        ("1_1_70","Foster commissary is clean all the time.","Positive comment"),
        ("1_1_71","More outside sales","Need more sale / coupons"),
        ("1_1_72","Nothing I can think of. Good staff!","Positive comment"),
        ("1_1_73","Price of lettuce is labeled per pound, but rings up per each","Overcharged on products"),
        ("1_1_74","Great Job! Thank you!","Positive comment"),
        ("1_1_75","More smaller cart.","Smaller carts required"),
        ("1_1_76","A few more options on the shelves and more abilities to restock.","Need more variety of Products options"),
        ("1_1_77","Keep up the great work!","Positive comment"),
        ("1_1_78","I was disappointed to find that flavored mineral water is only available in single bottles.  They are much healthier than the energy drinks which are available in abundance!","Need more variety of Products options"),
        ("1_1_79","Need the pork and oriental Ramen flavors back","Bring back discontinued products"),
        ("1_1_80","Someone needs to reevaluate the need to check IDs prior to using self checkout.  The ID is checked at the door and must be scanned by the register to ensure the patron is authorized to purchase at the commissary.  To have an employee require to check your ID prior to using the self checkout is unnecessary.  The cashiers do not check the ID but scan the card.  This places the employee in a very uncomfortable situation!!  Patrons are rude and don't understand the three factor verification required to use the commissary.  And when the employee's answer to the question of why is this required?, because they tell us we have to do it just puts them in a worse position.  My bank doesn't even require three factor authorization so why does self checkout at the commissary require it?  This is a waste of manpower and money paying an employee to unnecessarily check IDs.  Could someone please visit this practice and determine the cost involved.  Then weigh the cost involved in this ID check and see if it could be better used elsewhere.  If you believe this is a necessary practice, then inform users the necessity of the ID check so the employees are free from constant questions form patrons.","Decrease number times Check ID "),
        ("1_1_81","Blonde woman checking ID's at front entrance was very unfriendly. Several people greeted her with a Good Morning and she didn't respond. Where are her basic manners! Suggest some customer service training for her, or a new job! They could learn from the NEX employees! Always super nice and helpful!!","Associates need to be friendly towards customers"),
        ("1_1_82","Please correct the issue and hire more cashiers.","Increase check out lines and number cashiers"),
        ("1_1_83","Cashier is very good for the customer service","Positive comment"),
        ("1_1_84","There were a lot of empty shelves.  Don't know why!","Products not on shelf"),
        ("1_1_85","When I get home and I was putting away my groceries I looked at the receipt and saw that I was double charged for the one bottle of liquid bleach I brought. Obviously the bleach bottle was scanned twice so I paid 13.46 for one bleach instead of  $6.37. I also picked a boathouse mustard vintages dressing that was  suppose to 2.79 and I was charged $3.19. We live 1 hour and 1/2 from the base so we are not going to drive back to fix the overcharge, I'm sure it was accidental but still very annoying. Also the bagger bagged several  bags and we gave him a $3.00 tip. When we got home we realized that several bags just had 1-2 things in it and we actually consolidated and removed 4 bags. Did the bagger do this to get a bigger tip?","Overcharged on products"),
        ("1_1_86","The organization of the Commissary is the strangest of any store I shop in. Products are in multiple different places, like juice. Hummus is in at least three different places: snacks, vegetable section, and condiment/spread section. I alway take nearly and hour to shop here because products are not logically organized. All the dairy including milk should all be together. A store brand of an item is in one place but the brand name has its own stand.","Can't reach items on shelf"),
        ("1_1_87","The organization of the Commissary is the strangest of any store I shop in. Products are in multiple different places, like juice. Hummus is in at least three different places: snacks, vegetable section, and condiment/spread section. I alway take nearly and hour to shop here because products are not logically organized. All the dairy including milk should all be together. A store brand of an item is in one place but the brand name has its own stand.","Product placement needs improvement"),
        ("1_1_88","Found the prepackaged salad material to appear less than fresh, no lettuce in produce, otherwise produce section was well stocked and maintained. ","Need more variety of Produce"),
        ("1_1_89"," Out of the multi packs of facial tissues---the store in general was well stocked.","Products not on shelf"),
        ("1_1_90","More signs to direct to commissary.","Improve store signage "),
        ("1_1_91","Keep an eye on your check out line. When you have one cashier and six people in line, there's a problem.  This does not happen all the time, but when it does happen, nobody does anything to make it better.","Increase check out lines and number cashiers"),
        ("1_1_92","Get more self check-out lanes. There are 12 check out lines,out of which perhaps 2 have cashiers. There are 8 self check outs, that are always busy. Reduce manned lines to 6, use space for more self check out machines.","Increase check out lines and number cashiers"),
        ("1_1_93","I wish they carried everything I like to buy so I don't have to go to another grocery store.","Need more variety of Produce"),
        ("1_1_94","Lower prices!","Lower prices"),
        ("1_1_95","Some baggers have unethical attitude Watching our wallet or purse for the tips. Or grabbing the tips while we are still around that's rude","Associates need to be friendly towards customers"),
        ("1_1_96","There were only 2 registers open and a lengthy line waiting; the cashier explained many other cashiers had called in sick so it was a hectic morning.  The cashier (Pridyhara i think? #188) was very efficient and helpful.  Unfortunately, when i reached home, several items were missing.  I had loaded the car myself and am confident the cart was empty because i wheeled it across four lanes of parking to return the cart to the covered stand.  I did not make any stops on the way home.  My husband helped me search every bit of the car (a small Honda Fit) to no avail.  I called the commissary to ask if the missing bag was there, but was only able to record a message.  I drove back to the commissary and described the two baggers, one of whom was wearing a black hat with a silver chain and then a more petite lady whose number was 33.  The original cashier was there and tried to help, but needed to refer me to customer service.  Karen Francisco explained to me that no items had been reported but if anyone eventually found them she would call me.  I honestly don't think the bag of items ever made it into the cart; i had purchased $265 worth of groceries and it's difficult to both place items on the belt and watch bagging simultaneously to ensure everything is accounted for.  The line of questioning was awkward, as if i was somehow attempting to scam the items or had been inattentive when unloading.  I felt frustrated to have to make a second trip and spend an extra $36 to replace the missing items.  I retired after 28 years in the Navy and will continue to shop at the commissary, but it was a somewhat disappointing experience today.","Increase check out lines and number cashiers"),
        ("1_1_97","Tessie checking out customers was incredibly rude. Would not greet any customers nor hand them their receipt. Threw all grocery's onto the line for me to self bag.  Very rude","Associates need to be friendly towards customers"),
        ("1_1_98","baggers should be trained to pack meats with frozen items to keep them cold during warm weather. Would also like to see them earn at least min. wage rather than working for tips only. Also better availability for turkey bacon Oscar Mayer brand particularly.","Training for baggers"),
        ("1_1_99","I would like to see more organic produce options. Also would like to see more variety of flavors with the Jocko go fuel cans and Kodiak cake muffin/pancake cups.","Need more variety of Produce"),
        ("1_1_100","Having self checkout only on Mondays causes delays in line and makes it difficult when you have a lot of groceries. Having at least 1-2 checkout lines open during Mondays would help.","Increase check out lines and number cashiers"),
        ("1_1_101","iteem availability is often inconsistent","Products not on shelf"),
        ("1_1_102","Quality products that I habitually buy disappear never to return again.  These are everyday items like coffee maker descaler, jimmy dean sausage crumbles, certain salad dressings, unfrozen cooked chicken for salads, certain coffee brands.  More consistancy in products offered would be welcomed.","Products not on shelf"),
        ("1_1_103","Your stock was very low and on basic items.  I couldn't get cereals I use, baking power, nothing fancy just ordinary items that should be in stock.  Being open on Mondays but not restocking Monday night is big problem with this commissary","Products not on shelf"),
        ("1_1_104","Baggers missed an item, I paid for it but don't have it.","Missed placement item in bag"),
        ("1_1_105","a few of the items had lower prices than local store.  Cashier 107 was AWESOME!!","Local stores better prices"),
        ("1_1_106","I'm very happy with my shopping experience.","Great Store experience & Service"),
        ("1_1_107","More variety than local grocery stores ","Need more variety of Products options"),
        ("1_1_108"," lower priced!!!","Lower prices"),
        ("1_1_109","Need more Fresh produce to come in! Prices should be lower than off base!","Food / Produce not fresh"),
        ("1_1_110","Price check community.  Walmart beats most commissary prices except meats.","Local stores better prices"),
        ("1_1_111","A wider variety of products.","Need more variety of Products options"),
        ("1_1_112","Consistent on product availability","Products not on shelf"),
        ("1_1_113","More produce.","Need more variety of Produce"),
        ("1_1_114","Fill up most of the empty shelves. Order extra so items will not be NIS.","Products not on shelf"),
        ("1_1_115","Have more cashiers come to work cos some of us dont want to stand that long waiting. I waited in line longer than it took to shop!","Increase check out lines and number cashiers"),
        ("1_1_116","On my checking out experience the cashier Ms Joyce  help get 2 cases of water and a bag of dog food because it was heavy for me. I appreciate her effort helping older customers.I shopped at orote yesterday, I can not imagine the ichiban noodles limit is 2 each 1 case water and 2cases of water limit at Andersen. I feel sorry for a family of 5 or more limiting the family to drink water 8 ea. with children are athletic in school dad do p.t. everyday and sharing 2 ichiban for lunch or dinner.After typhoon only active duty and their dependent and essential personnel are allowed, the rest are discriminated. Didn't we suffered also, at least give 2 days that you think it is not busy for others next time anyway you limit everything.","Increase limit of Food Items"),
        ("1_1_117","Open commissar to all patrons all hours. There's no need to limit hours to AD. Naval Base Guam and AAFB should be aligned. One island one team concept should be taught to manager. Air Force and Navy need to be on the same page.","Store hours need increased"),
        ("1_1_118","Order different ice cream varieties","Need more icecream variety"),
        ("1_1_119","Variety of vegetables would be nice.","Need more variety of Produce"),
        ("1_1_120","Continue to hire people like Ms. Roslind she is a pillar to how people should be treated during their shopping experience.","Great Store experience & Service"),
        ("1_1_121","Not sure why you have rearranged the store, again.  We thought we know the location of the products that we purchased on the regular basis, now, we have to circle around the store to find the products that we normally buy.  The associate that we have asked about the location said that she does not know it either.  What a waste of money that you could have used to fight inflation.  Instead, you have paid someone to rearrange the store.  What a waste of money and human resources.  Why does the government always find ways to waste taxpayers money?","Improve store signage "),
        ("1_1_122","Great store","Great Store experience & Service"),
        ("1_1_123","Please Leave the light on in the restroom. I was scared when I had to use it because the light turned off on me while I was on the stall","Restrooms not clean & needs improvement"),
        ("1_1_124","The store was extremely HOT and the produce was suffering (along with every person) The quality was horrible but it was not the fault of anyone.  I will continue to travel to Redstone Arsenal from now on.","A/C not working properly"),
        ("1_1_125","Definitely different bread that doesn't have wheat since baby has wheat allergy.","Need more bakery products"),
        ("1_1_126","Due to inflation prices are very high. I can get similar or the same items off base at a cheaper price.","Local stores better prices"),
        ("1_1_127","I am becoming increasingly disappointed in empty shelves for products I wan to buy!  This time there wasn't any Morton Lite Salt or  Hebrew National Jumbo hotdogs.  My last shopping trip there were these two items, plus two others I couldn't get.  You have signs up about supply chain problems, but I can go to Safeway or Walmart and find those items on their shelves.  Sometimes it could be a month or more before and item returns to your shelves.  Then, on the next trip, they are gone again.","Products not on shelf"),
        ("1_1_128","Exceptional number of not in stock items","Products not on shelf"),
        ("1_1_129","Once again, this commissary falls far short of expectations.  insufficient cashiers, some substandard produce (obviously old and either wrinkling or turning brown).  These problems are on-going and do not seem to be addressed by management or higher authority.   Your commissary brand items are not that much of a savings over other/national brands, which is born out by direct price comparison in-store.  It is apparent that no attempt to improve the running of this commissary is being made.","Poor store shopping experience"),
        ("1_1_130","The cashier overcharged us several times by inputting the wrong produce, failing to recognize a markdown, saying a coupon applied when it hadn't.","Overcharged on products"),
        ("1_1_131","This commissary has continuous out of stock problems.  Tried to buy a roast chicken... normally carried at every grocery store.  Commissary had none and the deli rotisserie was not even running.  Had to change entire dinner plans based on that.   I got lucky and could use the express checkout..... 5 or 6 people waiting in line for an open checkout register.  This commissary continues to short staff the checkout registers and make patrons wait excessive times for an open register.","Products not on shelf"),
        ("1_1_132","The availability of international products","More available of international products"),
        ("1_1_133","Where is my Eddie? hes the only one that answers my questions and tells me whats going on. I have been asking for months about some of my favorite german foods. He is the only one that looked into the matter and the suppliers to get me the answers. please expand on the german foods section and stop shrinking it down.","Products not on shelf"),
        ("1_1_134","You need to put small (cans) lower so short Persons can reach them. Stop flip-flopping ,As it takes to much time to find where things are moved to. Some of have limited time to shop. It helps to know where things are. (I understand why you do it, as I worked retail for 24 yrs.) But food shopping is different.One wants to get in/out faster. Thank you!","Items not accessible "),
        ("1_1_135","More inventory","Need more variety of Products options"),
        ("1_1_136","More milk","Need more dairy products"),
        ("1_1_137","More outside sales","Need more sale / coupons"),
        ("1_1_138","More products","Need more variety of Products options"),
        ("1_1_139","More toilet paper and paper towel on sale","Need more sale / coupons"),
        ("1_1_140","The meats smell spoiled when you open the cryo packs, beef is extremely overpriced and the pa ks of pork are sold with so much fat that I am paying for â…” of fat and â…“ of actual meat. Cut selection is not that great and when you ask the butchers for a specific cut, they can never do it at the Foster Commissary, so now I have to drive to the Kadena commissary twice for the cut I want because it needs to be requested and then picked up on a different date but at least they cut it for you there.","Expiring Items "),
        ("1_1_141","When there was an item no longer in stock ","Products not on shelf"),
        ("1_1_142","You had the sushi I wanted","Need more variety of Products options"),
        ("1_1_143","The prices have become way too high, and most people are seeking elsewhere to buy locally farmers market","Local stores better prices"),
        ("1_1_144","I was disappointed to find that flavored mineral water is only available in single bottles.  They are much healthier than the energy drinks which are available in abundance!","Products not on shelf"),
        ("1_1_145","I wish associates had a clearer understand of coupons. I have multiple coupons to save money when we shop and they are regularly put in incorrect not saving me as much money as I would have saved. I had a coupon for a free product and instead of scanning the coupon the associate manually entered it for less than the amount off.","Coupons issues"),
        ("1_1_146","Produce needs to be at lower prices!  Why are strawberries and asparagus so expensive and they're in season?  I bought some asparagus several weeks ago for.99/lb and now it's over $3/lb.  Strawberries are too expensive.   We purchased packaged mushrooms and couldn't use them because of mold.  No Best Buy or expiration on the packaging.  I didn't feel comfortable returning them because I lost my receipt.","Lower prices"),
        ("1_1_147","Why are strawberries and asparagus so expensive and they're in season?  I bought some asparagus several weeks ago for.99/lb and now it's over $3/lb.  Strawberries are too expensive.   We purchased packaged mushrooms and couldn't use them because of mold.  No Best Buy or expiration on the packaging.  I didn't feel comfortable returning them because I lost my receipt.","Food / Produce not fresh"),
        ("1_1_148","I would like to have the option whether to print a receipt or not. It is a waste of paper for me, a waste of budget for the commissary if I have no use for it and a waste to the environment.","Print receipt option required"),
        ("1_1_149","Need the pork and oriental Ramen flavors back","Need more variety of Products options"),
        ("1_1_150","Sale price of item expired but display of sale price was never removed.  The wrong price was displayed over a week incorrectly. Would only offer the sale price for one item not all that I had in cart.  This caused an excessive time at checkout to resolve the store problem not mine.","Products missed priced"),
        ("1_1_151","The cleanliness of the men's bathroom was disgusting.  I walked in to find an elderly patron with a walker frustrated because he needed to use one of the stalls but both were filthy, clogged with waste and toilet paper and had water (or worse) on the floor.  There was loose trash on the sink countertop and the floor near the waste receptacle.  It was an utter embarrassment.  I immediately proceeded to the Customer Service window and let the employee know of the issue and she stated (with no enthusiasm or conviction) that she would let someone know.  I had to leave so could not verify if the issue was ever addressed.","Restrooms not clean & needs improvement"),
        ("1_1_152","First of all there were no carts at the entrence door as usual. ","No carts available"),
        ("1_1_153","Not enough baggers","Not enough baggers"),
        ("1_1_154","Someone needs to reevaluate the need to check IDs prior to using self checkout.  The ID is checked at the door and must be scanned by the register to ensure the patron is authorized to purchase at the commissary.  To have an employee require to check your ID prior to using the self checkout is unnecessary.  The cashiers do not check the ID but scan the card.  This places the employee in a very uncomfortable situation!!  Patrons are rude and don't understand the three factor verification required to use the commissary.  And when the employee's answer to the question of why is this required?, because they tell us we have to do it just puts them in a worse position.  My bank doesn't even require three factor authorization so why does self checkout at the commissary require it?  This is a waste of manpower and money paying an employee to unnecessarily check IDs.  Could someone please visit this practice and determine the cost involved.  Then weigh the cost involved in this ID check and see if it could be better used elsewhere.  If you believe this is a necessary practice, then inform users the necessity of the ID check so the employees are free from constant questions form patrons.","ID Verification process needs improvement"),
        ("1_1_155","The self-check out broke down after I'd entered all my groceries.  Two clerks were unable to get the machine to respond. Clerk 102 had to take my original receipt over to another regular, not self-checkout lane, to complete the transaction. I did not get the original receipt  I'm sure there is extra work for someone else at the store due to the breakdown. They handled it well, but I was stressed out because I didn't want my ice cream to melt. I'm sure you understand. I guess I wonder if I have to plan for extra delays in the future.  Also, question 25 on this survey does not accept any radio button. I had to skip to submit.","Self checkout issues"),
        ("1_1_156","better selection of cheeses","Need more variety cheese"),
        ("1_1_157","Better Store flow and signage","Improve store signage "),
        ("1_1_158","need more small carts.  more than half the time i come to the store they are all gone.  the past 7 times I have shopped I have been unable to get chocolate pie crust.  they have plain and graham cracker, but the chocolate has been out of stock for months?","Need more shopping carts"),
        ("1_1_159","seems a lot of the items are empty, noticed they allow civilians to shop... my husband put his life on line for this country.. this is one of few benefits we have left and now you allow civilians ... shelves have been emptier tried different time of day to see if it matters... it doesn't","Products not on shelf"),
        ("1_1_160","we have less choices of products... instead of 5 we may have 2-3","Need more variety of Products options"),
        ("1_1_161","Hamburger buns with an expiration of date of 25 June were mold. Had to return to store manager was very nice and accommodating. Purchased cheese at Deli when I first are no one was behind the counter so when I finished shopping I went back. ","Expiring Items "),
        ("1_1_162","Employee made it feel I was inconveniencing her and suggested I finish my shopping. Told her I had and she then proceeded to cut my cheese at slice thickness 5, my bad I didn't specify a level 2, but I got the impression it was quicker for her to do a 5. The only deli employee that I find courteous and pleasant all the time is the manager. There was an another one but I haven't seen her in a while.","Poor store shopping experience"),
        ("1_1_163","I was charged for, but did not receive one package of salmon. I've called the store twice but have not been able to speak with anyone about a refund for this item. I've left 2 voicemail messages. It's only $4.92 but the fact remains that I did not receive the product. I drive 30 miles roundtrip to the commissary & do not want to make a special trip to pick up the item.","Overcharged on products"),
        ("1_1_164","I am totally and thoroughly appalled at the piss pour condition of this commissary.  They were doing inventory and the store shelves were literally empty.  I travel 50 miles round trip to find such a disgusting situation.  ","Products not on shelf"),
        ("1_1_165","  I travel 50 miles round trip to find such a disgusting situation.  This is a routine and regular situation at this commissary and NO ONE seems to give a shit!!!!!!  You can complain and NOTHING IS EVER DONE TO CORRECT THE SITUATION.  The routine and continuous excuses are it's on the truck, the truck did not come, there is no one to stock the shelves.  If a regular store was run this way the would be out of business with this kind of nonsense.","Poor store shopping experience"),
        ("1_1_166","I asked customer service if they could order in Minute rice Jalapeno rice cups, I was told their supplier only has that type in a variety pack and they are not able to order it. I dropped packing with the bar code, and they could not. Bar code number 0 17400 22417 2","Products not on shelf"),
        ("1_1_167","I believe that the ladies restroom could use a little updating.  It is beginning to look a little shabby.","Restrooms not clean & needs improvement"),
        ("1_1_168","I dislike the name of the store brand Freedmanydom's Choice is taking away good brands that in cases were cheaper and better.","Dislike product name"),
        ("1_1_169","I find that the women's restroom could use some deep cleaning (toilet bowls very stained)","Restrooms not clean & needs improvement"),
        ("1_1_170","One cashier open for non-express orders, with no help in sight.  Cashier tried to get help, no one in office answered the phone.  I had to ask express cashier if she would take larger orders.  Customers with large orders do not want to use the self-checkout, as this is more of a hassle than it is worth.  The holding area is weighted, so you can not move anything, without having to call a cashier to assist.  ","Need more cashiers"),
        ("1_1_171"," The holding area is weighted, so you can not move anything, without having to call a cashier to assist.  This is not the first time that there has been only one cashier on duty, and no managers or assistants around to help with reducing the line.  It is unfortunate that this happens, as I believe this is why there are fewer people who now shop the commissary.  Items are out of stock for several weeks, i.e., graham cracker crumbs for example.  I find it hard to believe this is a national shortage.the express cashier was gracious enough to check me out, once I asked if she would.  It is not the cashier's fault; this is a direct result of poor management.","Poor store shopping experience"),
        ("1_1_172","Produce department seems more expensive than other stores I would like to see it more resonantly.","Local stores better prices"),
        ("1_1_173","Shaka ice tea","Need more variety of Products options"),
        ("1_1_174","The ladies bathroom is horrific. Needs to be remodeled and cleaner! There is never any hand soap to the right of the sinks.","Restrooms not clean & needs improvement"),
        ("1_1_175","We shopped on a Friday afternoon.  Only ONE clerk operated check-out line open!  As an older retiree, we don't do/like self check-out.  Express checker available, but our cart was full. All in our line had full carts and were older.  Check out was only bad part of our shopping trip!","Need more cashiers"),
        ("1_1_176","I was very disappointed at the majorly empty shelves in the bread section. ","Products not on shelf"),
        ("1_1_177"," I was appalled at the fact that in the bacon/sausage/pre-packaged deli meats, there were so many products that either were almost expired or were already past date!  And there were huge quantities of these meats that are bad--total waste of food.  Also frustrating that I couldn't get what I wanted because there were piles of the product that were not good to buy. And I even reached to the back to find products dated later than April 8, 2023 (with today being the 6th).  The self-checkout cashier I spoke to about this issue told me to fill out the survey and maybe something will change.  Because whoever deals with the ordering doesn't listen to those who have to take out all the expired products.","Expiring Items "),
        ("1_1_178","I would like the commissary to observe religious and patriotic holidays by closing the store or reducing the hours significantly.  Commissary patrons would rather allow others to enjoy their families than demand the commissary employees to work on a holiday.  We are adults and are capable or managing our time and needs.","Observe religious and patriotic holidays"),
        ("1_1_179","Living in state of Delaware there is no tax, so I still can't get used surcharge, which is now 5%.  That does cut into savings that are shown on the receipt.","Product tax on product"),
        ("1_1_180","Need better looking produce and more gluten free products.","Need better quality produce"),
        ("1_1_181","I rarely shop at the Commissary on a Monday. I have shopped here for years. I was previously told it was fine to complete WIC on a Monday. I even told other foster parents that they can purchase WIC from the Commissary on Mondays if necessary. After telling the cashier I had WIC, he asked me to wait for the other cashier. When he arrived, he did not come to the register, just talked to the cashier and stayed in self checkout. The original cashier I spoke to informed me that they would no longer allow WIC on Mondays. If this is the case, it needs to be made known. If he had told me this earlier, I would have purchased the formula with cash and went about my day. I could have came back another day to complete my WIC transaction.","Need more cashiers"),
        ("1_1_182","Cashier was rough with our items, basically tossing them on the belt as my daughter tried to quickly bag them. Seemed to be in a bad mood l, unpleasant.","Poor store shopping experience"),
        ("1_1_183","Not sure why your are forcing the highlighting on  the receipt. Savings really aren't happening, not when I pay a Surcharge on every item I purchase.  In Florida I don't pay sales tax on food items, so your savings are off.Nothing ever changes, and it seems like an EGO trip to me.","Poor store shopping experience"),
        ("1_1_184","The incomplete parking lot has to have numbers or letters added to the car lanes.","Parking lot signage"),
        ("1_1_185","I find prices too high on can goods, produce, dry goods, drinks, frozen, etc. ","Lower prices"),
        ("1_1_186"," I found non brand names are just as good or even better than brand names sold at Aldi or Lidl for less than half the prices at the commissary. The commissary does have better prices on meats, milk, eggs, and some bread that I go to the commissary for. Other items that I usually shop for at the commissary have been discontinued or no longer available. I also noticed that the commissary on an Army or Air Force base is cheaper and carry products that are not or no longer carried at a Navy Commissary.","Need more variety of Products options"),
        ("1_1_187","I found some of the hard to find items","Products not on shelf"),
        ("1_1_188","I try to keep my shopping items under the express checkout limit.On more than one occasion, I've waited while a customer checks out with an item count well over the limit.  Associates should redirect customers to the proper checkout.","Poor checkout experience"),
        ("1_1_189","I would like to see the express lanes come back","Need express lane"),
        ("1_1_190","The item I could not find was pomegranate seeds in a plastic container. I use them to make my breakfast. I also could not find the brand of honey and oats granola that I use to make my breakfast yogurt bow but I did find a suitable subsitiute.","Need more variety of Products options"),
        ("1_1_191","This particular store does not seem to carry ground Johnsonville sausage (mild or spicy Italian flavor) that I am able to find at little creek, Norfolk, and I was able to find in Guam.  They have the same brand but only in links which makes it harder to use for my cooking.  I'd love to see them start to carry the ground version of the product","Need more variety of Products options"),
        ("1_1_192","Two of the items I purchased, couldn't be read in the nearby stationary barcode reader. So I had to ask one of the cashiers for assistance. Get that fixed please.","Stationary barcode reader needs fixed"),
        ("1_1_193","Very disappointed.  Corn was advertised as 4 for $1.00. I was charged .69 cents and it was a holiday special. And charged $2.76. ","Overcharged on products"),
        ("1_1_194","Having the variety of Belvita bars I like","Need more variety of Products options"),
        ("1_1_195","I purchased items put on clearance due to the commissary discontinuing the products. The shelf price was listed as .80ea. I was charged $1.69 ea.","Overcharged on products"),
        ("1_1_196","I would like to see all attendants working self check out, responsive to customers needs and keeping scan area clean.","Need better checkout service"),
        ("1_1_197","The baggers were fooling around. Too lazy to make the effort to get the bags in a VAN....I guess it's just easier to pile them on top of each other. 1354 was our bagger","Need better checkout service"),
        ("1_1_198","Give me updates on special order items. I ordered some flood lights several weeks ago and have heard nothing.","Need to responsive to requests"),
        ("1_1_199","Have fresh meat readily available early Tuesday mornings during shopping for handicapped folks.","Improve meat selection"),
        ("1_1_200","I have been shopping at this store for 21 years. The management over the last few years has been decidedly worse and unresponsive. Products that I find at the commissary and enjoy suddenly disappear. I submit written requests asking for them to be returned and they never are and there's never any reply. Like laundry detergent. There must be seven different varieties of the most expensive laundry detergents, but only a single type of inexpensive detergent, and it is often out of stock. Same with juices and many other products. Thus commissary officer only carries the most expensive products generally. It is not a space issue. There must be 100 bottles of Tide detergent on display.","Products not on shelf"),
        ("1_1_201","I have been shopping at this store for 21 years. The management over the last few years has been decidedly worse and unresponsive. I submit written requests asking for them to be returned and they never are and there's never any reply. Mismanagement continually removes bargain priced products and replaces them with still more copies of more expensive products. Like laundry detergent. ","Poor store shopping experience"),
        ("1_1_202","I noticed it was warmer than normal, guess that's the overhead work fixing the airconditioner","A/C not working properly"),
        ("1_1_203","I was not delighted this trip.","Poor store shopping experience"),
        ("1_1_204","Keep merchanise stocked and consitent.","Products not on shelf"),
        ("1_1_205","Please ask the commissary officer to respond in writing to written requests for products. If they can no longer be procured for some reason, please provide that reason.","Need to responsive to requests"),
        ("1_1_206","They stock expired dairy all the time many items have expiration dates have brought to managers attention multiple times but keeps happening over and over would love to be able to buy all groceries today didn't even have a bag of potatoes or red peppers lettuce or fresh fruit what is going on? Last Nov we told them about expired evaporated milk on shelf 3 different times think you need some people that work there that cares and does there job manager included","Expiring Items "),
        ("1_1_207","Bathroom is falling apart and gross- old doors don't open all the time. A/C not working- hot in store- moving things around to where you have to hunt down","Restrooms not clean & needs improvement"),
        ("1_1_208","the produce is TERRIBLE. The tea is out of stock and many eggs were broken.","Food / Produce not fresh"),
        ("1_1_209","The produce is usually hit or miss as far as availability and quality.","Food / Produce not fresh"),
        ("1_1_210","would like to see better quality produce.","Food / Produce not fresh"),
        ("1_1_211","Prices should be on product or shelf. I picked up a very small straw cheese cake. I was very disappointed to findout at the register how much it was costing me to purchase. It the price was marked in plain view, I would not have purchased.","Products not on shelf"),
        ("1_1_212","Please keep product in stock!!!","Products not on shelf"),
        ("1_1_213","Pls refill items not in the shelves","Products not on shelf"),
        ("1_1_214","Produce selection could improve","Need more variety of Products options"),
        ("1_1_215","We were at the self checkout which sometimes has technical issues. There is a customer associate named Emily that has been vigilant in assisting myself and others on numerous occasions with her technical knowledge of systems she operates. Would like to make sure she is acknowledged for her hard work and customer service.","Self checkout issues"),
        ("1_1_216","I would like to see more Choice meats, rather that Select, at reasonable prices.  Also, maybe a few more coupons on the Commissary Rewards Card.  They always seem to be for the same products, month by month.  The price for name brand products still seems to be high.  If the prices dropped more, I would shop at the Commissary more often!","Need more variety of Products options"),
        ("1_1_217","Prices are the same as other groc stores in the area. Have more sales","Need more sale / coupons"),
        ("1_1_218","Why is there no buttermilk","Need more variety of Products options"),
        ("1_1_219","Would appreciate availability of deli salads and fresh fish.","Need more variety of Products options"),
        ("1_1_220","The produce section needs improvement. Overripe avocados, bruised potatoes, sad looking Brussels sprouts¦. Please try to improve quality!","Food / Produce not fresh"),
        ("1_1_221","More cleaning restroom,more friendly from cashiers.","Restrooms not clean & needs improvement"),
        ("1_1_222","the ice cream cake is $17.98 and was mushed.  ","Food / Produce not fresh"),
        ("1_1_223","The strawberries were rotten.  This commissary really has to do better with produce","Improve produce selection"),
        ("1_1_224","Associates are pleasant and professional.","Positive comment"),
        ("1_1_225","The signs for where to enter the check-out line need to be higher so you can see them.","Improve store signage "),
        ("1_1_226","Have more Mexican products","Need more variety of Products options"),
        ("1_1_227","I love how neat and tidy everything on the shelf is. The ice cream area needs more variety and product.","Need more variety of Products options"),
        ("1_1_228","I purchased a pack of (8) chicken thighs and they was bad","Food / Produce not fresh"),
        ("1_1_229","Too many times I see a price for an item get it and later at home I check the receipt and the price is higher than what I saw on the store shelves","Overcharged on products"),
        ("1_1_230","You program prices into the computer system at DECA level, but are not changed at the local level on the shelf.","Overcharged on products"),
        ("1_1_231","Get Fresher  produce.  Why are the asparagus and broccoli always old? The asian cabbage is always dry and ready to throw out. We can buy fresher fruit by the same brand at other grocery stores. The produce section is pretty bad. Been like that for a long time.  Do the Commissary buyers ask for the oldest fruits and vegetables? Do they get a discount for that? Do you just keep it in storage too long?  Somethng is not right!  System needs to be fixed!","Food / Produce not fresh"),
        ("1_1_232","Get fresher produce, although I know that sometimes its just not available.","Food / Produce not fresh"),
        ("1_1_233","more coupons","Need more sale / coupons"),
        ("1_1_234","There is no baking powder for sale and there hasn't been any baking powder for sale at the commissary for literally months.  This is not an exaggeration. How is this possible?  No baking powder stocked for months?? ","Products not on shelf"),
        ("1_1_235"," The quality of the steaks in the meat section is poor.","Improve meat selection"),
        ("1_1_236","Use to carry Taco Bell Fire sauce, Jimmy Dean Hashbrown Meat Lovers. Soda prices are too high compared to sales off base.","Bring back discontinued products"),
        ("1_1_237","With COVID-19 generally behind us, will the commissary bring back coffee available for customers near the store entrance as it was in the past?","Bring back discontinued products"),
        ("1_1_238","Bagger # 51 made very arrogant and insensitive comment about the cereal I purchased","Training for baggers"),
        ("1_1_239","I think the commissary is doing Fine job.","Positive comment"),
        ("1_1_240","Meat selection very low.","Improve meat selection"),
        ("1_1_241","Remove straps or damaged veggies from bins","Improve produce selection"),
        ("1_1_242","Addition of a Deli for boar's head lunch meats or other premium brands.","Improve meat selection"),
        ("1_1_243","Great Commissary!","Positive comment"),
        ("1_1_244","Paint the parking lot lines","Parking lot signage"),
        ("1_1_245","The produce is ALWAYS disappointing. ","Improve produce selection"),
        ("1_1_246","And other staple products were unavailable. I was told to go the McChord. ( Not convienent)","Increase produce selection"),
        ("1_1_247","Having a little bit more room to maneuver would be nice, I always seem to be squeezing around someone at the end of an isle or down the main lanes","Improve store signage "),
        ("1_1_248","I don't know why but the deli department is the worst ever.  I asked for a manager and he said there is no one there - he is the only one. I then brought this issue to the manager on duty who kindly agrees the meat was poor quality after showing it to her. She did offer to assist me but I declined and didn't want to deal with this deli any further. This was the second time I encountered this issue. As I was shopping the gentleman from the deli came up to me me and with hostility, tried to discuss the issue. WATCH your tape I was about to deck him","associates need to be friendly towards customers"),
        ("1_1_249","Love my commissary","Positive comment"),
        ("1_1_250","Cleaner restrooms","Restrooms not clean & needs improvement"),
        ("1_1_251","I get confused which door is an entrance/exit and how to get carts","Improve store signage "),
        ("1_1_252","Look at other competitors and what they charge. My family and I can get a cart full of food like fruit, produce, milk and other things for $80 to $100. While at the commissary I will get minimal to small amount of things and walk out paying 50-80 dollars. It's not the same as 7-10 years ago.","Lower prices"),
        ("1_1_253","When will there be email receipt availability?","Need email receipts"),
        ("1_1_254","You briefly carried a line of gluten free bread products that were abruptly replaced by Kings Hawaiian bread.  There are many people who need gluten free bread, and this particular one is the best, hard to find in stores, and expensive.  There is no rationale for a one-for-one replacement!","Need more bakery products"),
        ("1_1_255","Need to get back the following items: Dr. Praeger Spinach Little, Kale burgers, etc.  Red lentils.  Organic Valley cheese sticks.  Secondly, it would be great if the Commissary could get less expensive organic food.  If prices get lower, we will buy them at the Commissary.","Bring back discontinued products"),
        ("1_1_256","Carry more Decaf drinks and coffee.  Need to stock Milk Bonse soft chews.","Need more variety of Products options"),
        ("1_1_257","Rotisserie  chicken prices are higher then at Sam's club.","Lower prices"),
        ("1_1_258","Friendly staff, clean store","Positive comment"),
        ("1_1_259","Perhaps the associate assigned to work self checkout should actually man the self checkout so that when issues arise they can help or assist the customer when problems arise, instead of requiring the customer to stand around waiting for assistance with zero acknowledgment from the associate. When the associate assigned to self checkout was approached for assistance, she was unavailable to help.","associates need to be friendly towards customers"),
        ("1_1_260","Don't change anything","Positive comment"),
        ("1_1_261","Keep shelves well stocked.","Positive comment"),
    ]
    # testing de dup labels 
    save_data = [
        (int("111"), "Baggers are unfriendly", "Baggers unfriendly"),
        (int("111"), "Baggers are unfriendly, they never smile or greet customers", "Baggers unfriendly"),
        (int("112"), "Baggers are unfriendly, seem indifferent and cold", "Baggers unfriendly"),
        (int("113"), "Baggers lack any friendly interaction", "Baggers unfriendly"),
        (int("114"), "Baggers are unfriendly, there's no warmth from the baggers at checkout", "Baggers unfriendly"),
        (int("115"), "Baggers are unfriendly and they are unresponsive and distant", "Baggers unfriendly"),
        (int("116"), "Baggers don't engage with shoppers at all", "Baggers unfriendly"),
        (int("117"), "A total lack of courtesy from the bagging staff", "Baggers unfriendly"),
        (int("118"), "Baggers are consistently unfriendly and unhelpful", "Baggers unfriendly"),
        (int("119"), "Unfriendly baggers make the shopping experience less pleasant", "Baggers unfriendly"),
        (int("1110"), "The baggers' aloof demeanor is noticeable", "Baggers unfriendly"),
        (int("121"), "Prices are much higher than expected", "Price expensive"),
        (int("131"), "The price is high", "Price is high"),
        (int("131"), "The price is very high", "Price is high"),
        (int("122"), "The cost of products here is quite steep", "Price expensive"),
        (int("123"), "Found the prices to be excessively high", "Price expensive"),
        (int("124"), "Everything seems overpriced in this store", "Price expensive"),
        (int("125"), "The pricing here is not budget-friendly at all", "Price expensive"),
        (int("131"), "The price point is higher than average", "Price is high"),
        (int("132"), "Prices are significantly above what's reasonable", "Price is high"),
        (int("133"), "Noticed a high price tag on most items", "Price is high"),
        (int("134"), "Goods are sold at premium prices here", "Price is high"),
        (int("135"), "The pricing is on the higher side overall", "Price is high"),
        (int("141"), "Prices are higher compared to local markets", "Price higher than local"),
        (int("142"), "Local stores offer much better prices", "Price higher than local"),
        (int("143"), "Compared to nearby shops, prices here are elevated", "Price higher than local"),
        (int("151"), "The store lacks a welcoming atmosphere", "Store experience not good"),
        (int("152"), "Customer service needs significant improvement", "Store experience not good"),
        (int("153"), "Store layout is confusing and unorganized", "Store experience not good"),
        (int("154"), "Had a negative experience with the store staff", "Store experience not good"),
        (int("155"), "The overall shopping environment is not customer-friendly", "Store experience not good")
    ]

    save_data = [
        (int("111"), "Baggers are unfriendly", "Baggers unfriendly"),
        (int("112"), "Baggers are very unfriendly", "Baggers unfriendly"),
        (int("114"), "Baggers are very very unfriendly", "Baggers unhelpful"),
        (int("115"), "Baggers are infifferent and mean and not helpful", "Baggers unfriendly"),
        (int("116"), "Price is very high", "Price high"),
        (int("117"), "Price is high and expensive", "Price expensive"),
    ]
    

    #init_vector_db()
    #save_result = save_training_data(save_data, question_id)
    batch_size=30
    init_vector_db()
    for i in range(0, len(save_data), batch_size):
        batch = save_data[i : i + batch_size]
        print_debug("calling save_st with for " + str(i) + " times")
        save_training_data(batch, question_id,"unit_test")
        print_debug("save data in milvus done")
        save_st_model(batch, question_id)
        print_debug("save ST model is done")
    

# # Check if the correct number of command line arguments is provided
# if len(sys.argv) != 2:
#     print_debug("Invalid number of arguments. Please provide 'save' or 'search'.")
#     sys.exit(1)
# # Extract the operation from the command line argument
# operation = sys.argv[1]

# # Call the main function with the specified operation
# main(operation)
if __name__ == "__main__":
    main("unit_test_save")
    #main("save_trng")
    #main("unit_test_search")
"""
CREATE TABLE 

"""
