"""
cold start: 
topic 
split sentence
insight

topic -> cluster -> group topics 
insight -> collapse to similar insight if they belong to same topic

tie all of them 
original review id, original review, topic, split_sentence_id, split sentence, group_topic, group_insight

do sampling
split_sentence_id, sample split sentence, group_topic, group_insight

"""
import os
import openai
import json  # Import the JSON module to parse the JSON string
import time
import sys
from utils import print_debug, exec_sql

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# custom 
import utils
from utils import pretty_print_json

openai.api_key = "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"

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
llm_model="gpt-3.5-turbo"
llm_model = "gpt-4"
#llm_model="gpt-4-1106-preview"

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)

def call_openAI(system_content, user_content):
    # Combine the system and user content along with the JSON data to form the complete prompt
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    max_retries = 5
    temperature_val=0.0
    token_size_val=300
    model_name="gpt-3.5-turbo"
    model_name="gpt-4"
    #model_name="gpt-4-1106-preview"


    for attempt in range(max_retries):
        try:
            # Make a call to OpenAI's Chat API
            response = openai.ChatCompletion.create(
                model=model_name,  # Or the model of your choice
                messages=messages,
                temperature=temperature_val,
                max_tokens=token_size_val
            )

            # Extract and return the last message from the response
            # Convert the response to a JSON string
            response_str = response.choices[0].message['content']
            # Validate the JSON string
            response_json = json.loads(response_str)
            if isinstance(response_json, list):
                return response_str
            else:
                print(f"Response format is not a list. Retrying ({attempt+1}/{max_retries})...")
                time.sleep(1 + attempt)

        except json.JSONDecodeError:
            print(f"Failed to decode JSON response. Retrying ({attempt+1}/{max_retries})...")
            print(response_str)
            time.sleep(1 + attempt)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1 + attempt)

    print("Maximum retries reached. Unable to get a successful response.")
    return json.dumps([])

# Assume 'input_data' is the nested JSON array you provided
def extract_review_topics(input_data):
    # This dictionary will store the review_id as the key and a set of topics as the value
    review_topics = {}

    for review_set in input_data:
        for review in review_set:
            review_id = review['review_id']
            topic = review['topic']
            # Initialize a set for this review_id if not already present
            if review_id not in review_topics:
                review_topics[review_id] = set()
            # Add the topic to the set for this review_id
            review_topics[review_id].add(topic)

    # Now convert the dictionary into the required output format
    output = [[review_id, list(topics)] for review_id, topics in review_topics.items()]
    return output

def cluster_topics(context, topics):
    # Example specific topics and parent topics
    #parent_topics = ["Price", "Customer Service", "Product Variety", "Store Experience", "Promotions","Others"]

    # Update the user content with examples
    user_content = "I have a list of topics which you need to group each topic under a new parent topic. \
            Here is some context of the topics:" \
            f"\n{context}" \
            "Your task is to generate parent topic and then map the topic to it \
            The parent topics count should be <= 20% of the topics count.  i.e if there are 40 topics, \
            you should group them in <= 5 parent topics.  \
            Output the result should ONLY be JSON list exluding filler text in the following format. \
            [{'Parent Topic':<parent_topic1>,'Topics':[<topic1>,<topic2>]},'Parent Topic':<parent_topic2>:..]}]" + \
            f"\n\nSpecific Topics: {topics}"   
    
    system_content = "Process the provided lists and generate a nested \
        JSON structure mapping each specific topic to a corresponding parent topic."

    system_content = "Process the provided lists and generate a nested \
        JSON structure mapping each specific topic to a generated parent topic."

    # Convert lists to JSON for the prompt
    json_topics = json.dumps(topics)
    
    # Pass the data to function2
    response = call_openAI(system_content, user_content)
    response_list = json.loads(response)

    # Iterate through the list and print each item
    #for item in response_list:
    #    print(item)
    return response_list

def main(review_arr):
    review_id = review_arr[0]
    review = review_arr[1]
    
    # LLM chain exampple
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    zero_prompt = ChatPromptTemplate.from_template(
        "Your task is to find sentiment, its intensity for \
        each review from a list of reviews. \
        There must be only one sentiment per review \
        Possible sentiment values are: positive, negative, neutral \
        Possible sentiment intensites are : strong, mild, medium \
        \
        An examples of review text, sentiment are:  \
        review: 'We do not have any stores near by from our home \
        and we have no car, so we don't go to store and \
        shop at competitor store.  Though we know \
        the store has lot of products and produce \
        which we need.' \
        \
        Output for the above will be: negative, mild, medium  \
        where negative is sentiment, its intensity is medium and \
        \
        Now Find sentiment from the review :"
        "\n\n{review_text}"
    )
    chain_zero = LLMChain(llm=llm, prompt=zero_prompt, 
                        output_key="sentiment"
                    )
    
    first_prompt = ChatPromptTemplate.from_template(
        "Your task is to Find topics from the reviews. \
        A review can have multiple topics \
        An examples of review text and related topics are:  \
        review: We don't have any stores near by from our home \
        and we have no car, so we don't go to store and \
        shop at competitor store.  Though we know \
        the store has lot of products and produce \
        which we need. \
        topics for the above will be: Transportation issue, Variety of Produce  \
        \
        Now Find topics from the review:"
        "\n\n{review_text}"
    )
    #print(first_prompt)
    chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                        output_key="topics"
                    )
    # prompt template 2
    second_prompt = ChatPromptTemplate.from_template(
        "Split the review text in multiple subsets based on the topics. \
            Each subset can only have one topic. Try to minimize number of topics. \
            Ourput format will be a json array like topic: topic1 value, \
            split_sentence:sentence 1 value.., topic: topic2 ... \
            "
        "\n\nReview Text: {review_text} and topics: {topics}"
    )
    #print(second_prompt)
    chain_two = LLMChain(llm=llm,prompt=second_prompt, 
                        output_key="split_sentences")

    overall_chain = SequentialChain(
        chains=[chain_zero, chain_one, chain_two],
        input_variables=["review_text"],
        output_variables=["sentiment", "topics", "split_sentences"],
        verbose=True
    )
    result = overall_chain(review)
    #print(result)
    sentiment = result["sentiment"]
        
    pretty_print_json(result["topics"])
    #print("split_sentences_value\n" + result["split_sentences"])
    # Parse the JSON string into a list of dictionaries
    split_sentences_list = json.loads(result["split_sentences"])

    final_array=[]

    for item in split_sentences_list:
        final_array_current={}

        review_snippet = item["split_sentence"]
        topic_value = item["topic"]
        print(f"""review = "{review_snippet}" \n topic = "{topic_value}" """)
        if review_snippet == "" or topic_value == "":
            print("split sentence or topic is null, so skipping")
            continue
        
        review_topic = f"review = {review_snippet} and topic = {topic_value}"

        insight_schema = ResponseSchema(name="insight",
                description="extract actionable insight from the review. each review will have one insight and will be <= 5 words and output it as a string")
        reason_schema = ResponseSchema(name="reason",
                description="reason why you think that insight correspond to the review and output it as a string")
        
        response_schemas = [insight_schema, 
                            reason_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        #print(format_instructions)

        review_template = """\
        I am trying to extract actionable insight from review text. 
        Given a review text and topics associated with it, 
        you have to return a response with one actionable insight 
        that correspond to the survey and also specify the reason why you think that is the insight.   
        An insight should have 5 or less words. 
        Some sample themes could be , "Quality of food", "Price cheap", "Checkout line experience"

        For the following survey response, extract the following information:
        insight, reason

        review and topic text: {text} 
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template=review_template)
        print(prompt)
        
        messages = prompt.format_messages(text=review_topic,
                                        format_instructions=format_instructions)

        #print(messages[0].content)
        response = chat(messages)
        print("response chat " + str(response.content))
        output_dict = output_parser.parse(response.content)
        
        print(output_dict)
        insight=output_dict.get('insight')
        print("insight = " + insight)
        insight_reason = output_dict.get('reason')
        print("insight=" + insight + " reason:" + insight_reason)

        final_array_current={ 
            "review_id":review_id,
            "ans_txt": review,
            "sentiment": sentiment,
            "topic": topic_value,
            "split_sentence": review_snippet,
            "insight": insight,
            "insight_reason": insight_reason
        }
        final_array.append(final_array_current)

    
    pretty_print_json(final_array)
    return final_array

def merge_parent_topic(list1, list2):
    # Creating a dictionary from list2 for easy lookup
    topic_to_parent = {}
    for parent_topic_info in list2:
        for topic in parent_topic_info["Topics"]:
            topic_to_parent[topic] = parent_topic_info["Parent Topic"]

    # Adding "parent_topic" key-value pair before "topic" in each entry of list1
    for sublist in list1:
        for entry in sublist:
            topic = entry.get("topic")
            parent_topic = topic_to_parent.get(topic)

            # Inserting parent_topic before topic
            if parent_topic:
                # Find the position of "topic" in the keys list
                keys = list(entry.keys())
                topic_index = keys.index("topic")

                # Create a new ordered dictionary with parent_topic inserted before topic
                new_entry = {}
                for i, key in enumerate(keys):
                    if i == topic_index:
                        new_entry["parent_topic"] = parent_topic
                    new_entry[key] = entry[key]

                # Update the original entry
                entry.clear()
                entry.update(new_entry)

    return list1


def get_data_from_db(pipeline_job_id, limit_count):
    # SQL query to fetch the required data
    sql_fetch = f"""
    SELECT a.question, a.answer_id, a.ans_txt
    FROM ai.ai_all_process_llm a
    WHERE a.pipeline_job_id = {pipeline_job_id}
    -- AND answer_id=413212
    limit {limit_count}
    """

    # Execute the SQL query
    status, result = exec_sql(sql_fetch)
    if not status:
        return None, None

    # Initialize the variables
    question_txt = None
    reviews = []

    # Iterate over the result set
    for row in result:
        if question_txt is None:
            question_txt = row[0]  # Set the question text
        answer_id = row[1]
        ans_txt = row[2]
        reviews.append([answer_id, ans_txt])

    # Debug information
    if len(reviews) == 0:
        print_debug("** WARNING: No reviews fetched")
    else:
        print_debug("set of reviews to be processed:")
        print_debug(reviews)

    return question_txt, reviews

def save_records_to_db(results):
    exec_sql("truncate table ai.ai_all_process_llm_final")
    for record_list in results:
        for record in record_list:
            # Construct the column names and values for the INSERT statement
            columns = ', '.join(record.keys())
            values = ', '.join(["'" + str(value).replace("'", "''") + "'" for value in record.values()])

            # Construct the SQL INSERT statement
            sql_insert = f"INSERT INTO ai.ai_all_process_llm_final ({columns}) VALUES ({values})"

            # Execute the SQL statement
            status, result = exec_sql(sql_insert)
            if not status:
                print(f"Failed to insert record: {record}")
            else:
                print(f"Successfully inserted record: {record.get('review_id', 'N/A')}")


def unit_test(operation, pipeline_job_id, record_count):
    print("unit test, operation=" + operation + " job="+str(pipeline_job_id) + " record count =" + str(record_count))
    
    if operation == "unit_test":
        reviews = [
            [1,"The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
            [2,"As Discussed with Mr.Ram  Boss,\n\nPSR PDA TMR entry for all the towns should happen on 5th,10th,15th,20th..\n\nBenefits.\n1. Block Reports accessed through  PDA TMR Entr.   It Should reflect in TSE minet.\n2. Datas - Secondary happened in stockist town will be visible.\n3. TSE can understand  better knowledge about the Business & Teritorry & Action planning can be done.\n4.PSR  Efficiency\n5. Business Standards will increase by getting accuracy of data.\n\nThis can be done atleast for main Brands with respect to depot scenario.\n\nThanking You.\n\nRegards,\nH.Dhaamodharun"],
            [3,"The price of most of frozen items are more than local store. For example, price of streak is $59.99 where as local store selling for $45. Also A/C does not work most of the times"],
            [4,"The grocery store had a wide variety of produce.  But it was far way from our home by car.  So we decided not to go there"]
        ]
        reviews1 = [
            [1,"The crust on this pizza was perfectly crispy, and the cheese was melted to perfection. Definitely coming back for more!"],
            [2,"I was taken aback by the price – paying $30 for a medium pizza seems steep, even if it does taste good."],
            [3,"The pizza had a great flavor with fresh toppings and a delightful sauce. It was a bit pricey, but worth it for the quality."],
            [4,"Unfortunately, the pizza was quite disappointing – it was undercooked and bland. Definitely not worth the money."],
            [5,"I found the pizza to be exceptionally good, with a thin, crunchy crust and just the right amount of toppings."],
            [6,"While the pizza tasted fine, I was surprised by the bill. It's a bit much to charge $25 for a basic pepperoni."],
            [7,"This pizza place offers an excellent meal at a great price. You get a high-quality pizza without breaking the bank."],
            [8,"The gourmet pizza I ordered had an exciting blend of flavors that was truly satisfying, although the cost was on the higher side."],
            [9,"Sadly, the pizza was pretty tasteless and greasy. I had high hopes, but it just didn't live up to them."],
            [10,"For the price, I was expecting something special, but the pizza was just okay. Not sure I'll order from here again."]
        ]

        question = "How was your store experience? for grocery store"
        context = """
            The feedback is about a customer's experience in a grocery store. Themes in feedback can revolve around 
            product quality, staff behavior, store cleanliness, queue lengths at cashiers, product pricing, 
            discounts, offers, and other store-specific experiences.
        """

        context2 = f"related with grocery store feedback. \
            Example topics include 'Price is expensive', 'Customer service was poor', \
            'Great variety of products', etc.\
            Example of parent topics are 'Price', 'Customer Service', 'Product Variety', etc. \
        "
    else:
        context = """
            The feedbacks are from Marico India's sales people on question on cost reduction and increase productivity: 
            Question: 'Please provide all your IDEAS (BIG or SMALL) which will help Reduce 
            Cost or Increase Productivity across functions. Feel free to put in as many IDEAS as you wish...'
            Marico India produces hair oil, edible oil etc. The reviews has their jargon, acronyms etc. 
        """
        context2 = f"It is feedback from Marico India's sales people on question on cost reduction and increase productivity  \
            Example topics include 'Price is expensive', 'Customer service was poor', \
            'Great variety of products', etc.\
            Example of parent topics are 'Price', 'Customer Service', 'Product Variety', etc. \
        "
    
        print("\n calling get data from db")
        question, reviews = get_data_from_db(pipeline_job_id,record_count)
        print("number of reviews = " + str(len(reviews)))
        #pretty_print_json(reviews)
        #sys.exit(1)

    result_final = []
    for customer_review in reviews:
        print("\n\n--------- Review ---------------\n\n")
        print(customer_review)
        if len(customer_review[1]) <= 10:
            print("skipping review as very less size")
        else:
            print("-----------------\n")
            result_arr = main(customer_review)
            result_final.append(result_arr)
    
    #pretty_print_json(result_final)
    
    # Replace 'input_data' with your actual nested JSON array variable
    topics_list_with_id = extract_review_topics(result_final)
    #print("\n topics list == ")
    #pretty_print_json(topics_list_with_id)
    
    # Extracting just the topic texts from each sublist
    topic_texts = [topic for _, topics in topics_list_with_id for topic in topics]
    #pretty_print_json(topic_texts)

    topics_with_parents = cluster_topics(context2, topic_texts)
    print("cluster topics with parents output")
    pretty_print_json(topics_with_parents)
    #sys.exit()

    
    print("calling merge topics with parent")
    result_with_parent_topic = merge_parent_topic(result_final, topics_with_parents)
    pretty_print_json(result_with_parent_topic)
    
    if operation == "db":
        print("\n calling save_records_to_db")
        save_records_to_db(result_with_parent_topic)

if __name__ == "__main__":
    category=sys.argv[1]
    pipeline_job_id=sys.argv[2]
    record_count=sys.argv[3]

    if len(sys.argv) > 1:
        # The first argument in sys.argv[1:] will be your parameter
        unit_test(category, pipeline_job_id, record_count)
    else:
        print("No parameter provided. Running unit test with default settings.")
        unit_test(None)

"""


# Example usage
results = [{
    "review_id": 413207,
    "full_review_text": "Reducing Cost :In Marico we reimburse DD commission to stockist as part of redistribution expense, at present it ranges from .05% to .25%, and sitting in present banking scenerio, most of the bank participate online banking/RTGS facility with minimum charges, and evaluation can be done on this and  it definitely will reduce the charges that is presently borne by Marico.",
    "sentiment": "positive, strong, medium",
    "topic": "Reimbursement Policy",
    "split_sentence": "In Marico we reimburse DD commission to stockist as part of redistribution expense, at present it ranges from .05% to .25%.",
    "insight": "Review reimbursement percentages",
    "insight_reason": "The review mentions that the reimbursement percentages range from .05% to .25%. This suggests that there may be room for adjustment or standardization of these percentages."
}]

insert_into_database(results)

"""