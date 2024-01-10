import json
import sys
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    IndexType
)
from utils import print_debug, exec_sql, pretty_print_json
MILVUS_SIMILARITY_THRESHOLD = 0.8  # Define your threshold here

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

# comments 
"""
    This function ranks questions based on their similarity scores and matches with certain attributes.

    Parameters:
    - search_results: A list of search results from Milvus, where each result contains an ID and a similarity score.
    - input_question: The question input by the user, containing attributes like client_id, sub_vertical_id, and vertical_id.
    - master_questions: A list of master questions from the database, each containing details like question_id, client_id, etc.

    The function works as follows:
    1. Initialize an empty list to store ranked questions.
    
    2. Iterate over each result in the first element of search_results:
       a. Extract the question_id and the Milvus similarity score from the result.
       b. Check if the Milvus similarity score is above a predefined threshold.
    
    3. For each result passing the similarity threshold:
       a. Find the matching question from master_questions using the question_id.
       b. Initialize a scoring system and a details dictionary to store match information.
       
    4. Calculate the score based on matches with the input question's attributes:
       a. Add 3 points for a client_id match (highest priority).
       b. Add 2 points for a sub_vertical_id match (medium priority).
       c. Add 1 point for a vertical_id match (lowest priority).
       d. Update the details dictionary with match information for each attribute.
       
    5. Append the matching question with its calculated score and details to the ranked_questions list.
    
    6. Sort the ranked_questions list based on the total_weighted_score, with higher scores first.
    
    7. Assign a rank to each question in the sorted list, starting from 1.
    
    8. Return the list of ranked questions, each with its rank and score details.

    The returned list provides a ranked set of questions based on similarity and attribute matches, which can be used for further processing or display.
    """
def rank_questions(search_results, input_question, master_questions):
    ranked_questions = []

    for result in search_results[0]:
        question_id = result.id
        milvus_similarity_score = result.distance

        # Check if Milvus similarity score is above the threshold
        if milvus_similarity_score < MILVUS_SIMILARITY_THRESHOLD:
            matching_question = next((q for q in master_questions if q['question_id'] == question_id), None)
            if matching_question:
                details = {'milvus_similarity': f"{milvus_similarity_score:.2f}"}
                score = 0  # Initial score

                # Check for client, sub_vertical, and vertical matches
                if matching_question['client_id'] == input_question['client_id']:
                    score += 3  # Highest priority for client_id match
                    details['client'] = "match"
                else:
                    details['client'] = "no match"

                if matching_question['sub_vertical_id'] == input_question['sub_vertical_id']:
                    score += 2  # Next priority for sub_vertical match
                    details['sub_vertical'] = "match"
                else:
                    details['sub_vertical'] = "no match"

                if matching_question['vertical_id'] == input_question['vertical_id']:
                    score += 1  # Lowest priority for vertical match
                    details['vertical'] = "match"
                else:
                    details['vertical'] = "no match"

                matching_question['total_weighted_score'] = score
                matching_question['details'] = details
                ranked_questions.append(matching_question)

    # Sort questions based on the calculated score, higher score is better
    ranked_questions.sort(key=lambda q: q['total_weighted_score'], reverse=True)

    # Assign rank
    for i, question in enumerate(ranked_questions, start=1):
        question['rank'] = i

    return ranked_questions

def find_similar_questions(input_question, master_questions, collection_name, top_k=15):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    collection = Collection(name=collection_name)
    collection.load()

    input_embedding = model.encode(input_question['question_text'])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    search_results = collection.search([input_embedding], "embeddings", search_params, limit=top_k, output_fields=["question_id"])

    ranked_questions = rank_questions(search_results, input_question, master_questions)
    return ranked_questions[:top_k]

def insert_questions_into_milvus(master_questions, collection_name, drop_existing=False):
    # Connect to Milvus
    connections.connect()

    # Drop the collection if it exists and drop_existing is True
    if utility.has_collection(collection_name) and drop_existing:
        utility.drop_collection(collection_name)
        print(f"Existing collection '{collection_name}' dropped.")

    # Check if the collection exists, create if not
    if not utility.has_collection(collection_name):
        # Define the schema for the collection
        fields = [
            FieldSchema(name="question_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)  # Adjust dimension if needed
        ]
        schema = CollectionSchema(fields, description="Question Embeddings Collection")
        
        # Create the collection
        Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")

    # Initialize the model for embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode the questions
    embeddings = model.encode([q['question_text'] for q in master_questions])

    # Prepare question_id and embeddings for insertion
    question_ids = [q['question_id'] for q in master_questions]
    embedding_lists = [emb.tolist() for emb in embeddings]

    # Get the collection
    collection = Collection(name=collection_name)

    # Insert data into the collection
    collection.insert([question_ids, embedding_lists])

    # Attempt to create an index (will fail if the collection is loaded)
    try:
        # Define the index parameters
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }

        # Create an index on the embeddings field
        collection.create_index(field_name="embeddings", index_params=index_params)
        print(f"Index created for 'embeddings' field in '{collection_name}' collection.")
    except Exception as e:
        print(f"Failed to create index: {e}")
        # Optional: You can handle specific exceptions here if needed

    # Load the collection into memory
    collection.load()
    print(f"Collection '{collection_name}' loaded into memory.")

    print(f"Data inserted into '{collection_name}'.")

def get_data_from_db(limit=None):
    # SQL query
    sql = """
       SELECT
            a.question_id,
            a.question,
            e.industry_id AS vertical_id,
            e.specialization_id AS sub_vertical_id,
            d.client_id,
        MAX(sub.ans_count) AS max_ans_count
        FROM triestai.questions a
        JOIN (SELECT question_id, COUNT(ans_txt) AS ans_count FROM triestai.answers GROUP BY question_id) sub ON a.question_id = sub.question_id
        JOIN triestai.factors b ON a.factor_id = b.factor_id
        JOIN triestai.forms c ON c.root_factor_id = b.parent_id
        JOIN triestai.clients d ON d.client_id = c.client_id
        JOIN triestai.client_verticals e ON e.client_id = d.client_id
        WHERE a.que_type_id  in (5,6)
        GROUP BY a.question_id, a.question, e.industry_id, e.specialization_id, d.client_id
        ORDER BY e.industry_id, e.specialization_id, d.client_id, max_ans_count DESC
    """

    if limit is not None:
        sql += f" LIMIT {limit}"

    # Execute the query using the exec_sql utility
    status, results = exec_sql(sql)

    # Check if no records are found
    if not status:
        print("No records found.")
        return []

    # Convert the results to the desired Python array format
    master_questions = [
        {'question_id': row[0], 'question_text': row[1], 'vertical_id': row[2], 'sub_vertical_id': row[3], 'client_id': row[4]}
        for row in results
    ]

    return master_questions

def update_data_to_db(data, pipeline_job_id):
    # Extract the top 5 question texts for pipeline_output
    pipeline_output = json.dumps([item['question_id'] for item in sorted(data, key=lambda x: x['rank'])[:5]])

    # Convert the entire input data to JSON for pipeline_output_details
    pipeline_output_details = json.dumps(data)

    # SQL Update statement
    update_sql = """
        UPDATE ai_pipeline.ai_question_similarity 
        SET pipeline_output = %s , pipeline_output_details = %s, job_status='Processed'
        WHERE pipeline_job_id = %s
    """

    # Values for the SQL command
    values = (pipeline_output, pipeline_output_details, pipeline_job_id)

    # Execute the SQL command using exec_sql utility
    exec_sql(update_sql, values)
    print_debug(update_sql)
    print_debug("output details updated")

def main(operation, pipeline_job_id=None, question_id=None):
    if operation == "unit_test":
        # Get data from unit_test
        master_questions, input_question = unit_test()

        # Insert questions into Milvus and find similar questions
        init_vector_db()
        insert_questions_into_milvus(master_questions, 'tmp_question_similarity', True)
        top_questions = find_similar_questions(input_question, master_questions, 'tmp_question_similarity')
        
        # Print results for debugging
        print("Input question:", input_question['question_text'])
        #pretty_print_json(top_questions)

    elif operation == "similar_question_st":
        if question_id is None or pipeline_job_id is None:
            print("Error: question_id and pipeline_job_id are required for similar_question_st operation")
            return

        # Fetch master questions and prepare input question
        master_questions = get_data_from_db()  # Assuming get_data_from_db is defined
        input_question = next((q for q in master_questions if q['question_id'] == question_id), None)

        if input_question is None:
            print(f"Question with ID {question_id} not found in master questions.")
            return

        # Remove the input question from master_questions list
        master_questions = [q for q in master_questions if q['question_id'] != question_id]
        
        
        # Pipeline logic
        init_vector_db()
        insert_questions_into_milvus(master_questions, 'tmp_question_similarity', True)
        top_questions = find_similar_questions(input_question, master_questions, 'tmp_question_similarity')
        
        # Update database with results
        update_data_to_db(top_questions, pipeline_job_id)

    else:
        print("Invalid operation choice")
        return

    print_debug("input question= " + str(input_question))
    print()
    pretty_print_json(top_questions)


def unit_test():
    # Prepare the master questions and input question
    master_questions = [
        {'question_id': 1, 'question_text': 'what did you like about your store experience?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 2, 'question_text': 'How is your store expereince?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 11, 'question_text': 'What did you not like about store expereince?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 3, 'question_text': 'what did you like about your store experience?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c2'},
        {'question_id': 4, 'question_text': 'what did you like about your store experience?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv2', 'client_id': 'c3'},
        {'question_id': 5, 'question_text': 'what did you like about your store experience?', 'vertical_id': 'v2', 'sub_vertical_id': 'sv3', 'client_id': 'c4'},
        {'question_id': 6, 'question_text': 'Can I change my delivery address?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 7, 'question_text': 'What payment methods are accepted?','vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c6'},
        {'question_id': 8, 'question_text': 'How do I create an account?','vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 9, 'question_text': 'How do you like our international products?', 'vertical_id': 'v3', 'sub_vertical_id': 'sv2', 'client_id': 'c7'},
        {'question_id': 10, 'question_text': 'What is the return policy for sale items?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv2', 'client_id': 'c3'},
        {'question_id': 11, 'question_text': 'what brands you you wish to have in our store?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 12, 'question_text': 'Can you share what aspects of your shopping experience stood out?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 13, 'question_text': 'What were the positive highlights of your visit to our store?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 14, 'question_text': 'Could you describe what you enjoyed during your time at our store?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 15, 'question_text': 'What elements of your store visit did you find appealing?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'},
        {'question_id': 16, 'question_text': 'I am curious, what parts of your experience in our store were pleasing to you?', 'vertical_id': 'v1', 'sub_vertical_id': 'sv1', 'client_id': 'c1'}
    ]

    input_question = {
        'question_text': 'what did you like about your store experience?', 
        'client_id': 'c1', 
        'sub_vertical_id': 'sv1', 
        'vertical_id': 'v1'
    }

    return master_questions, input_question


if __name__ == "__main__":
    operation_arg = sys.argv[1] if len(sys.argv) > 1 else None
    question_id_arg = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    pipeline_job_id_arg = sys.argv[3] if len(sys.argv) > 3 else None
    if operation_arg not in ["unit_test", "similar_question_st"]:
        print("Invalid operation. Choose 'unit_test' or 'similar_question_st'")
    elif operation_arg == "similar_question_st" and (question_id_arg is None or pipeline_job_id_arg is None):
        print("Question ID and Pipeline Job ID are required for 'similar_question_st'")
    else:
        print_debug("qid = " + str(question_id_arg) + " pipeline=" + str(pipeline_job_id_arg) + " operation =" + operation_arg)    
        main(operation_arg, pipeline_job_id_arg, question_id_arg)

