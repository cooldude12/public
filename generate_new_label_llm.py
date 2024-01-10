import json
import time
import openai
import time
import sys
import utils
from utils import print_debug, exec_sql

openai.api_key = "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"

def is_valid_format(json_str):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print_debug("load json failed")
        return False

    if not isinstance(data, list):
        print_debug("isInstance(list) json failed")
        return False

    for text in data:
        if not isinstance(text, str):
            return False
        if len(text.split()) >= 10:
            return False
    return True

def generate_label(prompt, temperature=0, model="gpt-3.5-turbo", max_retries=5, **kwargs):
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
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error occurred: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 3:
                time.sleep(1)
            elif attempt < max_retries - 2:
                time.sleep(2)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                print("cound not produce output after max retries")
                return []

def process_review_with_retry(prompt, temperature=0, model="gpt-3.5-turbo", max_retries=5):
    for attempt in range(max_retries):
        response = generate_label(prompt, temperature, model)
        if is_valid_format(response):
            return response
        else:
            if attempt == max_retries - 1:
                print(f"Invalid response on the last attempt. Response: {response}")
            else:
                print(f"Invalid response. Retrying... ({attempt+1}/{max_retries})")
    return []

def generate_insight(prompt):
    # Try to process the review with the given prompt
    #print_debug(prompt)
    response = process_review_with_retry(prompt)
    if not response:
        print(f"Failed to process the review based on the given prompt: {prompt}\n")
        return None
    else:
        #print(f"Processed prompt: {prompt}\nResponse: {response}\n")
        return response

def get_reviews(pipeline_job_id=0):
    # Retrieve the search data array
    print_debug("getting input data")
    sql = f"""
        SELECT 
            a.id as original_id,
            a.question_id,
            a.answer_id,
            IF(ISNULL(q.question), "N/A", q.question) AS question_text,
            b.pipeline_input as answer_text
        FROM
            ai_pipeline.ai_new_label_llm a
        LEFT JOIN JSON_TABLE(pipeline_input, '$[*]'
            COLUMNS (pipeline_input VARCHAR(150) PATH '$')) b ON TRUE
        LEFT JOIN triestai.questions q ON a.question_id = q.question_id
        WHERE a.job_status='Scheduled'
        AND a.pipeline_job_id={pipeline_job_id}
        """
    
    status, result = exec_sql(sql)
    if not status:
        return
    print_debug("input data lenghts = " + str(result))

    data = []
    for row in result:
        original_id = row[0]
        question_id = row[1]
        answer_id = row[2]
        question_text = row[3]
        answer_text = row[4]
        data.append([original_id, question_id, answer_id, question_text, answer_text])

    print(data)
    return data
     
def save_labels(data, pipeline_job_id=0):
    delete_query = f"DELETE FROM ai.ai_new_label_llm where pipeline_job_id={pipeline_job_id}"
    exec_sql(delete_query)

    for row in data:
        original_id = row[0]
        question_id = row[1]
        answer_id = row[2]
        question_text = row[3]
        answer_text = row[4]
        json_labels = row[5]
        
        insert_query = "INSERT INTO ai.ai_new_label_llm (original_id, pipeline_job_id, question_id, answer_id, \
                 question_text, answer_text, pipeline_output, job_status) \
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        values = (original_id, pipeline_job_id, question_id, answer_id, question_text, answer_text, json_labels, 'Processed')

        # delete existing records
        exec_sql(insert_query, values)

# update source tbl
def update_input_data_job_status(pipeline_job_id):
    
    sql_update = f"""
        UPDATE ai_pipeline.ai_new_label_llm a JOIN (
            SELECT original_id,
                answer_id,
                pipeline_output AS json_output
                FROM ai.ai_new_label_llm
                WHERE pipeline_job_id={pipeline_job_id}
                GROUP BY original_id, answer_id
            ) b  
            ON a.answer_id=b.answer_id 
            AND a.id = b.original_id  
            AND a.pipeline_job_id={pipeline_job_id}
            SET a.pipeline_output = b.json_output, 
                job_status='Processed'
    """
    exec_sql(sql_update)

#main
def main(operation, pipeline_job_id=0):
    print_debug("operation passed to main " + operation)
    if operation == "unit_test":
        reviews = unit_test()
    elif operation == "generate_new_label":
        reviews = get_reviews(pipeline_job_id)
        print_debug("review count = " + str(len(reviews)))
    else:
        print_debug("Wrong option passed to main")
        sys.exit(0)

    final_output = []
    for review in reviews:
        original_id, question_id, answer_id, question_text, review_text = review
        # prepare the prompt
        prompt = f"""
       
        You are an AI language model, and your primary objective is to correctly understand the 
        sentiment of the feedback and generate insights based on that sentiment. Here are your guidelines:

        1. Determine if the feedback is positive, negative, or neutral.
        2. If the feedback is positive, generate a positive insight that reflects the content. 
           It's crucial not to infer negative actionable insights from positive feedback.
           if feedback is this "Staff is alaways amazing!!" , label should be "good staff" 
           and NOT "Improve staff quality".  But if the feedback is negative like "Staff is rude and lack knoweledge"
           then label could be "Improve staff quality".  
        3. Negative feedback should lead to actionable insights, highlighting areas for improvement.
        4. Neutral feedback might not require any actionable insights. Focus on providing an accurate description if possible.
        5. You can produce multiple insights for a single review, but no more than 3.
        6. Negative insights should be prioritized over others.
        7. Keep each insight concise, with no more than 9 words.
        8. Non-text elements in the feedback, such as emojis or emoticons, can provide sentiment context. Ensure you interpret these correctly.
        9. A line break in the feedback might indicate a new theme or point. Consider each segment separately.
        10. If the feedback has an associated question, factor in the sentiment of that question. For example:
            Positive Question: "Why did you like the pizza?" 
            Feedback: "Because of the price." 
            Expected Insight: "Good price."

            Negative Question: "What didn't you like about the pizza?" 
            Feedback: "The price." 
            Expected Insight: "Price too high."
        11. It's possible a review doesn't have an associated question. In such cases, rely entirely on the feedback.
        12. At least one insight should be generated, if possible. However, it's okay not to have second or third insights.
        13. If you can't determine any insight from the feedback, label it as "Unknown."
        14. Insights should be ordered by their significance, with more actionable or impactful insights appearing first.

        Given the question: {question_text}
        And the feedback: {review_text}

        Please generate the appropriate insights in JSON format: ["<insight1>", "<insight2>", "<insight3>"] on a single line.

        """
        
        # generate label
        response = generate_insight(prompt)
        if response is not None and review_text is not None:
            print_debug(review_text + "\n" + response)

        # check if the response is not None and it is a valid JSON
        if response and check_format(response):
            # append to final output
            #final_output.append(response)
            final_output.append([original_id, question_id, answer_id, question_text, review_text, response])
        else:
            print("Failed to generate insight or the generated insight is not a valid JSON")

    print_debug(final_output)
    if operation == "generate_new_label":
        save_labels(final_output, pipeline_job_id)
        print_debug("saved labels in the DB")
        update_input_data_job_status(pipeline_job_id)

    #return final_output

def check_format(response):
    # try to load the response into json
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        print("Failed to parse response into JSON:", str(e))        
        return False

    # check if insights is a list
    if not isinstance(response, list):
        return False

    # check if all elements in the list are strings
    if not all(isinstance(insight, str) for insight in response):
        return False

    return True

def unit_test():
    # list of question_id, answer_id, question_text, answer_text
    reviews = [
        [1, 1, 'How was your pizza?', 'The pizza was great, but it arrived late. The customer service was not that great either.'],
        [2, 2, 'What do you think about our new Margherita?', 'The Margherita pizza was fresh and had lots of cheese, but the crust was a little hard.'],
        [3, 3, 'What do you think of our service?', 'Service was quick and efficient, but your location is too far.'],
        [4, 4, 'How was the ambiance of our restaurant?', 'The ambiance was nice, but it was a little too noisy.'],
        [5, 5, 'What are your thoughts on our vegetarian options?', 'The vegetarian options were diverse, but could use some more creativity.'],
        [6, 6, 'What do you think about our gluten-free options?', 'The gluten-free pizza was surprisingly good, but the options are very limited.'],
        [7, 7, 'How do you like our gourmet pizzas?', 'The gourmet pizza was delicious, but it was a bit expensive.'],
        [8, 8, 'What do you think about our delivery speed?', 'The delivery speed was decent, but the pizza arrived cold.'],
        [9, 9, 'How do you rate our customer service?', 'Customer service was poor and the wait time was too long.'],
        [10, 10, 'What do you think about our dessert pizzas?', 'The dessert pizza was too sweet and did not have enough toppings.'],
        [11, 11, 'How was your overall experience with our service?', 'The overall experience was satisfactory, but there is room for improvement.'],
        [12, 12, 'What do you think of our pizza dough?', 'The pizza dough was nice and thin, but lacked flavor.'],
        [13, 13, 'How do you rate our toppings?', 'The toppings were fresh and abundant, but lacked variety.'],
        [14, 14, 'What do you think about our menu?', 'The menu has a wide variety, but it can be a bit confusing.'],
        [15, 15, 'How do you rate the cleanliness of our restaurant?', 'The restaurant was clean, but the bathrooms were not.'],
        [16, 16, 'What do you think about our drink options?', 'The drink options were good, but lacked non-alcoholic choices.'],
        [17, 17, 'How was the temperature of your food?', 'The food was well-cooked, but it could have been hotter.'],
        [18, 18, 'How do you rate our wait staff?', 'The wait staff was friendly, but they were not very attentive.'],
        [19, 19, 'What do you think of our special offers?', 'The special offers are great, but they are not very well advertised.'],
        [20, 20, 'What do you think about our pricing?', 'The pricing is fair, but the portion sizes could be larger.']
    ]
    reviews1 = [
        [1, 1, 'How was your pizza?', 'The pizza was great, but it arrived late. The customer service was not that great either.'],
        [2, 2, 'What do you think about our new Margherita?', 'The Margherita pizza was fresh and had lots of cheese, but the crust was a little hard.'],
        [3, 3, 'What do you think of our service?', 'Service was quick and efficient, but your location is too far.']
     ]
    reviews = [
        [11, 1, 1, 'How was your pizza?', 'The pizza was great, but it arrived late. The customer service was not that great either.'],
        [12, 2, 2, 'What do you think about our new Margherita?', 'The Margherita pizza was fresh and had lots of cheese, but the crust was a little hard.'],
        [13, 3, 3, 'What do you think of our service?', 'Service was quick and efficient, but your location is too far.'],
        [14, 4, 4, 'How was the ambiance of our restaurant?', 'The ambiance was nice, but it was a little too noisy.'],
    ]
    return reviews

if __name__ == "__main__":
    main("unit_test")

"""
extract data : 
 SELECT
    JSON_UNQUOTE(JSON_EXTRACT(pipeline_input, '$[0]')) AS feedback,
    JSON_UNQUOTE(JSON_EXTRACT(pipeline_output, '$[0]')) AS col1,
    JSON_UNQUOTE(JSON_EXTRACT(pipeline_output, '$[1]')) AS col2,
    JSON_UNQUOTE(JSON_EXTRACT(pipeline_output, '$[2]')) AS col3,
    a.*
from ai_new_label_llm a where pipeline_job_id=42 ;
select * from ai_pipeline.ai_save_training_data where pipeline_job_id=15;

promt old 
 You are an AI language model. 
        Your task to do the following actions. 
        1. generate actionable insights from a review.  But note that a review may have an insight but 
           that may not be actioanble. Typically that is for a positive review.  For example: 
           if feedback is this "Staff is alaways amazing!!" , label should be "good staff" 
           and NOT "Improve staff quality".  But if the feedback is negative like "Staff is rude and lack knoweledge"
           then label could be "Improve staff quality".   
        2. The insights can have multiple themes, like product defective, delivery issue, demand issue. 
        3. An example of actionable insight is: if the review says, "I liked the pizza, but the crust was hard"
        the insight will be , "Fix the hard crust"
        4. The number of insights should be maximum up to 3 for the review
        5. If a review has negative items, prioritize those insights
        6. Each insight should be <= 9 words
        7. The original review can have non-substantive elements [i.e non text] 
        like emojis, emoticons, filler words like "N/A".  
        8. if those emojis have good or bad meaning underlying, you should consider 
        that for generating insights.  
        9. there could be line breaks in review text.  line breaks often associated with a new theme. 
        10. Question text may have a positive or negative theme. generate the label keeping the sentiment
            of the question in mind. Here are two examples
            positve question theme: Why you liked the pizza" 
            Review: "Price" 
            Expected Label: "Good price" 

            Negative question theme: Why you did not like the pizza" 
            Review: "Price" 
            Expected Label: "Bad price"
        11. A review may not have an associated question.  
        12. It is ok not to have insight2, insight3, but try to have at least insight1. 
        13. If you cannot generate insight, then just put "unknown" for insight
        14. Order the insights on their strengths.  Give higher weight on insight that is more actionable.

        The question asked was: {question_text}
        The response given was: {review_text}

        Please generate the insights.

        The output should be a well-formatted JSON. 
        The format will be like this: ["<insight1>", "<insight2>", "<insight3>"]
        and will be in a single line.

"""