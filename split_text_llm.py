"""
split text pipeline. 
sample test data: 

truncate table ai_pipeline_jobs;
insert into ai_pipeline_jobs (user_id,client_id, study_id,question_id, pipeline_name, pipeline_id,
    dt_created, job_status, input_params) values 
    (1,1,1,3,'split_text',1, CURRENT_TIMESTAMP, 'scheduled',null);
update ai_pipeline_jobs set job_status='Processed';
update ai_pipeline_jobs set job_status='Scheduled' where pipeline_name='split_text' ; 
update ai_pipeline.ai_split_text set pipeline_job_id=1,label=null where question_id=3 ;
    
-- insert source data    
delete from ai_pipeline.ai_split_text where question_id=3;
INSERT INTO ai_pipeline.ai_split_text (`question_id`, `answer_id`, `question_text`, `answer_text`)
VALUES 
(3, 11, 'Your pizza dining experience?','The pizza was delicious but it was a bit cold when it arrived, and the delivery was late.'),
(3, 12, 'Your pizza dining experience?','The pizza tasted bad. It had too much cheese, and it was too salty for me.'),

update ai_split_text set question_text='How was your dining experience?' where question_id=3;
update ai_split_text set pipeline_job_id=1 where question_id=3;
commit;
commit;

how to run it:
>cd ~/triestai-backend/pipelines 
>python3 run_job_3.py

sample output: 
input data answer text: The pizza was delicious but it was a bit cold when it arrived, and the delivery was late.
generate split text label: ["The pizza was delicious but it was a bit cold when it arrived.", "The delivery was late."]

output tbl: ai_pipeline.ai_split_text

to run unit test: 
>cd ~/triestai-backend/pipelines 
>python3 split_feedback.py 

it will run unit_test() function.  you can modify test data in that funciton
"""

import json
import time
import openai
import time
import sys
import utils
from utils import print_debug, exec_sql, display_output
import os 
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate split feedback using OpenAI ChatCompletion API
def generate_split_feedback(prompt, temperature=0.8, model="gpt-3.5-turbo", max_retries=5, **kwargs):
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
            feedback_subsets = json.loads(response.choices[0].message.content)
            #feedback_subsets = json.loads(response.choices[0].message.content)
            # convert the Python list into a string with double quotes
            feedback_subsets = json.dumps(feedback_subsets)
            print_debug(feedback_subsets)
            return feedback_subsets
        
        except Exception as e:
            print(f"Error occurred: {e}. Retrying... ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("Failed to produce output after max retries")
                return []

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
    
# Function to save data to the database

def save_data_db(data, pipeline_job_id):
    # delete existing records
    delete_query = f"DELETE FROM ai.ai_split_text WHERE pipeline_job_id={pipeline_job_id}"
    exec_sql(delete_query)

    for row in data:
        split_text_id = row[0]
        question_id = row[1]
        answer_id = row[2]
        question_text = row[3]
        answer_text = row[4]
        json_labels = row[5]
        if answer_text is not None:
            print_debug("answer: " + answer_text + " label: " + json_labels)

        insert_query = "INSERT INTO ai.ai_split_text \
                (split_text_id, pipeline_job_id, question_id, answer_id, \
                question_text, answer_text, label) \
                VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (split_text_id, pipeline_job_id, question_id, answer_id, question_text, answer_text, json_labels)
    
        exec_sql(insert_query, values)
    
    # update the CRM ai db with the labels
    # update_query_crm = """UPDATE ai_pipeline.ai_split_text_llm a join ai.ai_split_text b 
    #             ON a.pipeline_job_id=b.pipeline_job_id 
    #             AND a.answer_id=b.answer_id 
    #             SET a.label=b.label,
    #             a.job_status='Processed'
    #             """    
    update_query_crm = """
        UPDATE ai_pipeline.ai_split_text_llm a join (
            SELECT split_text_id, pipeline_job_id,question_id,answer_id, 
                concat("[",(group_concat(concat('"',label,'"'))),"]") as label
            FROM ai.ai_split_text 
            group by split_text_id
        ) b 
        ON a.pipeline_job_id=b.pipeline_job_id 
        AND a.answer_id=b.answer_id 
        SET a.pipeline_output = REPLACE(b.label, '↵', '\n'),
        a.job_status='Processed'                
    """
    update_query_crm = f"""
        UPDATE ai_pipeline.ai_split_text_llm a  
        JOIN ai.ai_split_text b 
        ON a.pipeline_job_id=b.pipeline_job_id 
        AND a.answer_id=b.answer_id 
        SET a.pipeline_output = b.label,
        a.job_status='Processed' 
        AND  a.pipeline_job_id={pipeline_job_id}              
    """
    exec_sql(update_query_crm)

# Function to retrieve data for split
def get_data_for_split(pipeline_job_id):
    # Retrieve the search data array
    sql_delete="TRUNCATE TABLE ai.ai_split_text"
    sql_insert= f"""
            INSERT INTO ai.ai_split_text (
            split_text_id,
            pipeline_job_id,
            question_id,
            answer_id,
            question_text,
            answer_text,
            label,
            job_status
        )
        SELECT
            a.id,
            a.pipeline_job_id,
            a.question_id,
            a.answer_id,
            'Please share your review?', -- generic question text 
            b.pipeline_input,
            NULL,
            job_status
        FROM
            ai_pipeline.ai_split_text_llm a
        LEFT OUTER JOIN json_table(
            a.pipeline_input,
            '$[*]' COLUMNS (pipeline_input VARCHAR(150) PATH '$')
        ) b ON TRUE
        WHERE a.pipeline_job_id={pipeline_job_id}
        AND a.job_status='Scheduled'
        ;
    """
    exec_sql(sql_delete)
    exec_sql(sql_insert)
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

def main(operation,pipeline_job_id=0):
    print_debug("in split_feedback main(), operation=" + operation + " pipleine_id=" + str(pipeline_job_id))
    context = 'Here is a survey feedback' # this is a place holder/cathc all
    if operation == "unit_test":
        context, reviews = unit_test()
    elif operation == "split_text":
        reviews = get_data_for_split(pipeline_job_id)
    else:
        print_debug("invalid option")
        sys.exit(1)

    final_output = []
    if reviews is not None:
        for review in reviews:
            split_text_id, question_id, answer_id, question_text, review_text = review
            
            prompt = f"""
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

            print_debug(prompt)
            print_debug(review_text)
            response = generate_split_feedback(prompt)
            print_debug("response = " + str(response))
            if response and check_format(response):
                final_output.append([split_text_id, question_id, answer_id, question_text, review_text, response])
            else:
                print_debug("Failed to generate split feedback or the generated feedback is not a valid JSON")
        
        print_debug("***** Final Output ***** ")
        print_debug(final_output)
    else:
        print_debug("** WARNING: No answers to process")

    if operation == "split_text":
        save_data_db(final_output, pipeline_job_id)
        return True
    
    return final_output

# Your list of reviews
def unit_test():
    context = ''

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
    context = """
        The feedback is about a customer's experience in a grocery store. Themes in feedback can revolve around 
        product quality, staff behavior, store cleanliness, queue lengths at cashiers, product pricing, 
        discounts, offers, and other store-specific experiences.
    """
    return context, reviews

if __name__ == "__main__":
    operation = "unit_test"
    output = main(operation)
    display_output(output)

### 

"""
            You are an AI language model. Your task is to split a feedback if the feedback has multiple themes. 
            Each subset of the feedback should have only one theme. A subset can have one or more sentences.
            If the entire review has one theme, then it should NOT split into sentences. But it has multiple 
            themes, you should NOT have only one set, but multiple sets
            
            For example, if the feedback is: "The pizza was delicious, but the delivery was super late. 
            The delivery guy arrived two hours late", the subsets will be "The pizza was delicious" and  
            "The delivery was super late. The delivery guy arrived two hours late".

            A second example that has same theme, where it should NOT split the feedback. 
            "The pizza was horrible.  It was cold and very less toppings".  The output will be only one set
            "The pizza was horrible.  It was cold and very less toppings".

            Other things to take care of for the label
            1. sice a label represents only one theme, it should not have numbering or bullet points
            2. should not have emojis.  if emoji or similar non-ascii chars are present at source, 
               the emotion need to be converted to text
            
            You can also look at the question text if present to decide if there is one theme or multiple
            thems in the review. 
            
            Here is the input data for feedback:
            The question asked was: {question_text}
            The feedback given was: {review_text}

            Please split the above feedback into subsets according to the themes.
            The output should be a well-formatted JSON as ["<subset11>", "<subset12>", "<subset3>"] 
            and will be in a single line.
           
            You are an AI language model. Your primary task is to intelligently split customer feedback based on distinct themes. Follow the guidelines below:

            One Theme, One Set: Each subset of the feedback should represent a singular theme. If there's a distinct shift in theme within a feedback, split it into separate sets.

            No Split for Singular Themes: If the entire feedback revolves around a singular theme, it remains intact. However, if there are multiple themes, you must segregate them.

            Detailed Examples:

            Multi-theme Feedback: "The pizza was delicious, but the delivery was super late. The delivery person arrived two hours behind schedule." This feedback has two themes:
            The quality of the pizza.
            The tardiness of the delivery.
            Single-theme Feedback: "The pizza was undercooked and cold when delivered." This feedback has one theme: The unsatisfactory quality of the pizza.
            Formatting Requirements:

            Avoid bullet points or numbering.
            Convert emojis or non-ASCII characters to descriptive text.
            Context Consideration: Take cues from the accompanying question text (if provided) to better contextualize the feedback's theme(s).

            Input Data Format:

            Question: {question_text}
            Feedback: {review_text}
            Based on the above, extract the themes and format your output as a clean, one-line JSON array: ["<theme1>", "<theme2>", ...]
            
"""

"""

The prices in the store are too expensive, and the quality of the produce is terrible.', 
'["The prices in the store are too expensive", "the quality of the produce is terrible"]']

'The air conditioning in the store was not working, and it was sweltering inside.', 
'["The air conditioning in the store was not working, and it was sweltering inside."]']

'The baggers at the store were not friendly at all, and they mishandled my groceries.', 
'["The baggers at the store were not friendly at all they mishandled my groceries."]']

'The cashier line at the store was incredibly long, and I had to wait for ages to check out.', 
'["The cashier line at the store was incredibly long, and I had to wait for ages to check out."]']

"The prices of products at the store are exorbitant, and I won't be shopping here again.", 
'["The prices of products at the store are exorbitant", "I won\'t be shopping here again."]']

"""