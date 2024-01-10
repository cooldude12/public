"""

"""

import json
import time
import openai
import time
import sys
import utils
from utils import print_debug, exec_sql, display_output

openai.api_key = "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"

# Function to generate split feedback using OpenAI ChatCompletion API
def call_openAI(prompt, temperature=0.8, model="gpt-3.5-turbo", max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                **kwargs,
            )
            # split the response and format it to a list
            #feedback_subsets = [feedback.strip(' ".,') for feedback in response.choices[0].message.content.strip().split('\n')]
            response_subsets = json.loads(response.choices[0].message.content)
            #feedback_subsets = json.loads(response.choices[0].message.content)
            # convert the Python list into a string with double quotes
            response_subsets = json.dumps(response_subsets)
            #print_debug(feedback_subsets)
            return response_subsets
        
        except Exception as e:
            print(f"Error occurred: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("Failed to produce output after max retries")
                return []

def call_Llama2(prompt, temperature=0.8, model="gpt-3.5-turbo", max_retries=5, **kwargs):
    return None

def call_VertexAI(prompt, temperature=0.8, model="gpt-3.5-turbo", max_retries=5, **kwargs):
    return None


# Function to check the format of generated split feedback
def check_format(response):
    return True
    try:
        # try to load the response into json
        # if it fails, it is not valid json format
        json_feedback = json.loads(response)
        return all(isinstance(subset, str) for subset in json_feedback.values())
    except json.JSONDecodeError:
        return False
    
# Function to retrieve data for split
def get_data_for_split(pipeline_job_id):
   
    sql_fetch = f"SELECT ai.split_text_id,ai.question_id, \
             ai.answer_id, ai.question_text, ai.answer_text \
            FROM ai.ai_split_text ai \
            WHERE pipeline_job_id={pipeline_job_id}"

    status, result = exec_sql(sql_fetch)
    if not status:
        return
    
    data = []
    for row in result:
        split_text_id = row[0]
        question_id = row[1]
        answer_id = row[2]
        question_text = row[3]
        answer_text = row[4]
        data.append([split_text_id, question_id, answer_id, question_text, answer_text])

    if len(data) == 0:
        print_debug("** WARNING: No answers fetched")
    else:
        print_debug("set of answers to be processed by the ML model:")
        print_debug(data)
        return data
    return

# Main function to execute the pipeline
import json

def format_json_array(json_array):
    formatted_json = json.dumps(json_array, indent=4)
    return formatted_json

def get_topics(context, example, reviews):
    output=[]
    for row in reviews:
        id, review = row
        print()
        print("Review--------------------------\n"+review)     
        prompt_topic = f"""
            I am trying to classify survey responses based on certain topics. 
            Given a survey response, you have to return a response with a list of topics 
            that correspond to the survey and also specify the reason why you think it matches.   
            A topic should have 2 or more words.  
            So instead of "price", it should be "Price expensive" etc. 
            Some sample themes could be , "Quality of food", "Price cheap", "Checkout line experience"

            A response can also have multiple topics. 

            The output should be a  json array with 3 elements
            "topic", "reason"
            
            A topics example is: 
            {example}

            Now do the the above for the following  Survey response: 
            the survey is for {context}.  

            survey response:
            {review}
        """
        response = call_openAI(prompt_topic)
        print(response)
        output.append(response)
        #response = call_Llama2(prompt)
        #topics = [item["topic"] for item in response["topics"]]
    return output

    
def split_feedback(context, question_text, topic_list, reviews):
    
    final_output = []
    if reviews is not None:
        for review in reviews:
            split_text_id, question_id, answer_id, review_text = review
            
           
            prompt = f"""
                I am trying to classify survey responses based on certain themes. 
                The reviews are of this subject: 
                {context}
                
                The possible themes are: 
                {topic_list}
                
                Given a survey response, you have to return a response with the 
                details of the themes that match the survey. 
                Also give me the relevant phrase from the response. 
                
                The response should be a formatted json array with labels "topic", and "phrase"
                Each line should have one topic and associated reason and phrase. 
                for example if there are two response records output format will be: 
                [ 
                    'topic1': topic1
                    'phrase': phrase1
                ],     
                [ 
                    'topic': topic2
                    'phrase': phrase2
                ]
                Following are the guardrails. 
                1. All the phrases should be subset of the sentence.
                No phrase from orgiginal survey survey response to be dropped i.e if all the phrases
                are added they should make the survey response.
                2. If there are emojis, they should be omitted. 
                
                Survey response:
                {review_text}

            """
            
            #print_debug(prompt)
            print("\n ----------------------------------------")
            print(review_text)
            response = call_openAI(prompt)
            #response = call_Llama2(prompt)

            #response_formatted = format_json_array(response)
            print(response)
            if response and check_format(response):
                final_output.append([split_text_id, question_id, answer_id, question_text, review_text, response])
            else:
                print_debug("Failed to generate split feedback or the generated feedback is not a valid JSON")
        
        #print_debug("***** Final Output ***** ")
        #print_debug(final_output)
    else:
        print_debug("** WARNING: No answers to process")

    
    return final_output

def get_topic_list_from_db():
    list = "price, quality of food, delivery issue, checkout experience."
    return list

def get_sample_list_from_vectordb():
    return None

# Your list of reviews
def unit_test_topic():
    example = f"""
        review: 
        "The cashier line at the store was incredibly long, and I had to wait for ages to check out. 
        Prices of the goods were expensive as well" 
        for the above review, the topics will be: "checkout experience", "price""
    """
    context = "Grocery Store Reviews"
    reviews = [
        [1, "The grocery store had a wide variety of produce.  But it was far way from our home by car.  So we decided not to go there"],
        [2, "The price of most of frozen items are more than local store. For example, price of streak is $59.99 where as local store selling for $45. Also A/C does not work most of the times"]
    ]
    full_response = get_topics(context, example, reviews)
    #print(topic_list)
    #print(full_response)

def unit_test_split_feedback():
    
    reviews = [
        [1, 1, 1,  "The cashier line at the store was incredibly long, and I had to wait for ages to check out. Prices of the goods were expensive as well."],
        [1, 1, 1,  "The price of most of frozen items are more than local store. For example, price of streak is $59.99 where as local store selling for $45. Also A/C does not work most of the times"],
        [2, 2, 1, "The grocery store had a wide variety of produce.  But it was far way from our home by car.  So we decided not to go there"]
    ]
    question = "How was your store experience? for grocery store"
    context = """
        The feedback is about a customer's experience in a grocery store. Themes in feedback can revolve around 
        product quality, staff behavior, store cleanliness, queue lengths at cashiers, product pricing, 
        discounts, offers, and other store-specific experiences.
    """
    topic_list = get_topic_list_from_db() 
    example_samples = get_sample_list_from_vectordb() 

    split_feedback(context, question, topic_list, reviews)
    

if __name__ == "__main__":
    #unit_test_topic()
    unit_test_split_feedback()
   

"""
 prompt = f
                You are an AI language model. Your primary task is to intelligently segment customer feedback based on distinct themes. Follow the guidelines below:
                Direct Extraction: Extract and segment the original feedback text without altering or adding any words. The output should be direct fragments from the input, and you should not generate new content.
                Distinct Themes: Each segment of the feedback should represent a singular theme. If there's a distinct shift in theme within a feedback, split it into separate sets. However, be mindful of subtle connections in the feedback. For example, a feedback like 'I hated the pizza as the crust was too hard' has a single theme, even though it mentions both the pizza and the crust.
                No Split for Strongly Connected Themes: If the entire feedback revolves around a closely related or singular theme, it remains intact. For example, 'I hated the pizza because the crust was too hard' should not be split, as it revolves around the overall dissatisfaction with the pizza.

                Formatting Requirements:

                Do not use bullet points or numbering.
                Convert emojis or non-ASCII characters to descriptive text.
                Context Consideration: Use the accompanying question text (if provided) to better contextualize the feedback's theme(s).

                Input Data Context: {context} 

                It's essential to understand the interconnectedness of the themes and ensure that feedback about 
                closely related experiences isn't unnecessarily split.
                Format: 
                Question: {question_text}
                Feedback: {review_text}
                Based on the above, extract the themes and format your output as a clean, one-line JSON array: ["<theme1>", "<theme2>", ...]

                Examples:

                Input: "The pizza was delicious but the delivery was super late."
                Output: ["The pizza was delicious", "The delivery was super late."]

                Input: "I hated the pizza because the crust was too hard."
                Output: ["I hated the pizza because the crust was too hard."]

                Input: "The cashier line was  long, I waited for a long time."
                Output: ["The cashier line was  long, I waited for a long time"]

                Input: "baggers were bad, did not help at all."
                Output: ["baggers were bad, did not help at all"]

                ''
                Use the examples to guide your segmentation. If the feedback elements are strongly interrelated, keep them unified.    

"""
"""
 reviews = [
    [11,1, 1, "How was our pizza dining expereince?", "Pizza was delicious but the delivery was super late. The delivery guy came 2 hrs late."],
    [12, 1, 2, "How was our pizza dining expereince?", "Pizza tasted horrible. There were hardly any toppings and the cheese was salty."],
    [13, 1, 3, "How was our pizza dining expereince?", "Pizza tasted tasty and warm. it also has good cheese and toppings"],
    [14, 1, 4, "How was our pizza dining expereince?", "I loved the pizza but the curst was hard"],
    [15, 1, 5, "How was our pizza dining expereince?", "I hated the pizza as the curst was too hard"],
    [16, 1, 6, "How was our pizza dining expereince?", "I liked the pizza but the delibery was super late"],
    [17, 1, 7, "How was our pizza dining expereince?", "High price\n  No branding \n No Taste"]
    ]
    
    reviews = [
    [1, 1, 1, "How was your store experience? for grocery", "The prices in the store are too expensive, and the quality of the produce is terrible."],
    [2, 2, 1,  "How was your store experience? for grocery", "The air conditioning in the store was not working, and it was sweltering inside."],
    [3, 3, 1, "How was your store experience? for grocery", "The baggers at the store were not friendly at all, and they mishandled my groceries."],
    [4, 4, 1, "How was your store experience? for grocery", "The cashier line at the store was incredibly long, and I had to wait for ages to check out."],
    [5, 5, 1, "How was your store experience? for grocery", "The prices of products at the store are exorbitant, and I won't be shopping here again."],
    ]

    reviews1 = [
    [6, 1, 2, "How is your experience with the software?", "Using this software is a nightmare. It's so difficult to navigate, and I can never find what I need. On top of that, the login process rarely works, leaving me frustrated."],
    [7, 2, 2, "How is your experience with the software?", "The software is painfully slow. It takes ages to load anything, and trying to get work done is a real struggle. Plus, it's missing critical features that would make my job so much easier."],
    [8, 3, 2, "How is your experience with the software?", "I'm fed up with this software. It's incredibly hard to use, and I constantly run into login issues. It's a productivity killer."],
    [9, 4, 2, "How is your experience with the software?", "This software is a real time-waster. It's so slow that I spend more time waiting than actually getting work done. And it's missing important features that are essential for my tasks."],
    [10, 5, 2,"How is your experience with the software?", "I can't stand using this software. It's a hassle to navigate, and it takes forever to complete even simple tasks. Plus, the login problems are a constant headache."],
    ]
"""
