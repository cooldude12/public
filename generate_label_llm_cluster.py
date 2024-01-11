import openai
from collections import defaultdict
import json
import time
import openai
import time
import sys
import utils
from utils import print_debug, exec_sql
import os 

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_prompt(training_data):
    training_data_prompt = "\n"
    for feedback, label in training_data:
        training_data_prompt += f"- Feedback: {feedback}\n  Insight: {label}\n"
    

    # prepare the prompt
    prompt = f"""
    You are an AI language model. Your task is to perform the following actions:
    Your task is to identify the main themes or labels from the given feedback about a pizza service. 
    Below are some common themes found in past feedback:

        - Good Taste: Feedback that praises the taste of the pizza.
        Examples: "The pizza was delicious", "Loved the taste!"

        - Bad Taste: Feedback where the customer didn't like the taste.
        Examples: "Pizza was stale", "Didn't enjoy the pizza"

    Note , it is ok to have more than one themes in a feedback. So in that case you can generate multiple insights (aka: labels). 
    Using these themes, your goal is:

    1. Given a set of training data and test feedback, determine the similarity of test data with training data.
    2. Match the feedback to the most relevant themes. However don't be restricted by these themes. If you find a 
       theme in the feedback that is not listed, create a new, specific label or insight for it.   
    3. An insight should be actionable , it should have valuable content in it so that user gets value. For example if the feedback is: 
       "I liked the pizza, taste was great, cheese was yummy, toppings had lots of yummy vegatables.  
        Though crust was little hard, overall it was good expereince. I also found the price quite higher than other neighborhood stores"

       good labels:  "pizza Toppings had tasty vegatbles"   or "Fix the crust", "price higher than local competition"
       bad labels: "tasty pizza", "good toppings", "price high"

    4. If the test feedback is similar to any training instance, apply the relevant labels from the training data to the test data.
    5. A single feedback can have multiple themes and can therefore be assigned multiple labels, up to a maximum of 5 labels.
    6. Each label can have up to 2-5 words. Avoid creating labels that contradict each other.
    7. A single feedback might have multiple labels. Prioritize the labels based on the feedback's emphasis. do NOT be forced
       to create 5 labels.  Note that, 5 labels is the maximum.  if there is only one or thwo themes in the feedback, 
       just create 1 or 2 labels. Order the labels based on their relevance scores, with the most relevant label appearing first.
    8. Consider non-textual elements in the feedback, such as emojis and emoticons. Interpret their sentiment (positive, negative, neutral) and consider them when assigning labels.
    9. Feedback text might contain line breaks, which might signify a shift in theme or topic. Consider each line separately when determining its theme.
    10. The question posed might have its own sentiment. While assigning labels, factor in the sentiment of the question.  input for 
       test data will have question text and also feedback given.

    For example:
    - Positive question theme: "Why did you like the pizza?" 
        Review: "Price" 
        Expected Label: "Good price"

    - Negative question theme: "Why didn't you like the pizza?" 
        Review: "Price" 
        Expected Label: "High price"

    9. Some feedback might not have an associated question. In such cases, rely solely on the feedback text to assign labels.
    10. If a feedback doesn't match any training instance closely, label it as "Unknown."
    
    Here are some feedbacks and their corresponding insights:
    {training_data_prompt}

    The output should be a well-formatted JSON. The format will be like this: ["<label1>", "<label2>", "<label3>", "<label4>", "<label5>"] and should appear in a single line.
   
    """
   
    return prompt

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

def get_labels_for_feedbacks(test_data, prompt):
    results = []
    counter = 0
    prompt_final=""

    for data in test_data:
        counter = counter + 1 
        if counter > 10:
            sys.exit()

        q_id = data[0]
        a_id = data[1]
        q_text = data[2]
        a_text = data[3]
        
        prompt_final = prompt + f"""
        Given the above instructions, Please generate the labels with this input:

            question_text: {q_text}
            feedback_given: {a_text}
        """
        #print_debug("prompt:\n" + prompt_final)

        response_text = generate_label(prompt_final, temperature=0.5) 
        
        original_labels = eval(response_text)  # Convert the returned string to a Python list
        # Map each original label to its closest cluster
        clustered_labels = [cluster_labels(label) for label in original_labels]
        clustered_labels = [label for label in clustered_labels if label]  # Filter out any None values

        print_debug("answer_text: " + a_text + "\nlabels original :" + str(original_labels) + "\ncluster:" + str(clustered_labels))

        results.append({
            'question_id': q_id,
            'answer_id': a_id,
            'question_text': q_text,
            'answer_text': a_text,
            'original_labels': original_labels,
            'clustered_labels': clustered_labels
        })

    return results

def cluster_labels(labels):
    clusters = {
        "good taste": [],
        "bad taste": [],
        "reasonable price": [],
        "expensive price": [],
        "delivery late": [],
        "bad customer service": []
    }
    # for feedback, label in labels:
    #     for keyword in clusters:
    #         if keyword in label:
    #             clusters[keyword].append(feedback)
    #             break  # Assuming each feedback fits into one cluster for simplicity.
    # return clusters

    for cluster, keywords in clusters.items():
        for keyword in keywords:
            if keyword in labels:
                return cluster
    return None  # if no cluster matches


def generate_output(test_data, labels, clusters):
    output = []
    for idx, (_, _, _, feedback) in enumerate(test_data):
        clustered_label = next((k for k, v in clusters.items() if feedback in v), None)
        original_label = labels[idx][1]
        score = 1.0  # You may wish to incorporate a scoring mechanism later
        output.append((test_data[idx][0], test_data[idx][1], test_data[idx][2], feedback, clustered_label, original_label, score))
    return output

def get_training_data():
    training_data = [
        ("The pizza was delicious and full of flavor!", "good taste"),
        ("Delivery took forever and the pizza was cold when it arrived.", "delivery late, bad taste"),
        ("I think it's a bit pricey for the size of the pizza.", "price expensive"),
        ("Customer service was unhelpful when I called about a wrong order.", "bad customer service"),
        ("The crust was crispy and the toppings were fresh.", "good taste"),
        ("I wasn't impressed with the bland taste.", "bad taste"),
        ("Considering the quality, the price was quite reasonable.", "good taste, reasonable price"),
        ("Delivery was prompt and the pizza was still hot!", "good taste, fast delivery"),
        ("I had a hard time communicating with their customer service.", "bad customer service"),
        ("The pepperoni pizza was too oily for my liking.", "bad taste"),
        ("Despite the late delivery, the pizza still tasted great.", "delivery late, good taste"),
        ("I found a hair in my pizza. Terrible experience!", "bad taste, bad hygiene"),
        ("I love their vegetarian options. So flavorful!", "good taste"),
        ("The pizza arrived squished in the box. Not a pleasant sight.", "bad presentation"),
        ("The cheese burst pizza is worth every penny!", "good taste, reasonable price"),
        ("I didn't expect such a gourmet pizza at this price.", "good taste, reasonable price"),
        ("The pizza was cold and tasted like cardboard.", "bad taste"),
        ("Delivery was quick but they forgot the dips.", "fast delivery, incomplete order"),
        ("The new spicy flavor is amazing!", "good taste"),
        ("It's the best pizza place in town, but slightly expensive.", "good taste, price expensive")
    ]
    return training_data

def get_test_data():
    test_data = [
    (1, 1, "How did you like our pizza?", "It tasted just perfect. Reminded me of Italy!"),
    (2, 2, "How did you like our pizza?", "I waited for more than an hour. Very slow delivery."),
    (3, 3, "How did you like our pizza?", "Honestly, I've tasted better. It was a bit bland."),
    (4, 4, "How did you like our pizza?", "The pizza was amazing but it was a bit overpriced."),
    (5, 5, "How did you like our pizza?", "I called customer service twice and no one picked up."),
    (6, 6, "How did you like our pizza?", "Loved the spicy pepperoni. It had the perfect kick!"),
    (7, 7, "How did you like our pizza?", "The vegan pizza toppings were fresh and delicious!"),
    (8, 8, "How did you like our pizza?", "The box was damaged and the pizza was all over the place."),
    (9, 9, "How did you like our pizza?", "I wish there were more cheese, but it was decent."),
    (10, 10, "How did you like our pizza?", "Please introduce a low-carb or keto option."),
    (11, 11, "How did you like our pizza?", "The BBQ Chicken pizza was a little too sweet for me."),
    (12, 12, "How did you like our pizza?", "Delivery was super fast. Very pleased."),
    (13, 13, "How did you like our pizza?", "The new dessert pizza was surprisingly good! Would order again."),
    (14, 14, "How did you like our pizza?", "The crust was too thick and chewy, not my favorite."),
    (15, 15, "How did you like our pizza?", "Probably not ordering again. I've had much better elsewhere."), 
    (16, 16, "How did you like our pizza?", "Your Margherita is as authentic as it gets. Reminds me of Naples."),
    (17, 17, "How did you like our pizza?", "I had a very friendly delivery person. Give him my regards!"),
    (18, 18, "How did you like our pizza?", "Your website was a bit hard to navigate. Had trouble placing an order."),
    (19, 19, "How did you like our pizza?", "The mushroom and truffle pizza was a delight!"),
    (20, 20, "How did you like our pizza?", "My kids love your cheesy bread. It's always a hit."),
    (21, 21, "How did you like our pizza?", "I'd recommend offering more vegetarian options."),
    (22, 22, "How did you like our pizza?", "I found a hair in my pizza. Very disappointed."),
    (23, 23, "How did you like our pizza?", "Your Hawaiian pizza is the best in town. Perfect pineapple to ham ratio."),
    (24, 24, "How did you like our pizza?", "The olives tasted a bit off. Everything else was good."),
    (25, 25, "How did you like our pizza?", "Impressed with the hygiene and safety measures during delivery."),
    (26, 26, "How did you like our pizza?", "The pizza was cold by the time it arrived."),
    (27, 27, "How did you like our pizza?", "The garlic dipping sauce was so good!"),
    (28, 28, "How did you like our pizza?", "More vegan cheese options would be great."),
    (29, 29, "How did you like our pizza?", "The tomato sauce was a bit too tangy for my liking."),
    (30, 30, "How did you like our pizza?", "Great crust, ample toppings, and arrived hot."),
    (31, 31, "How did you like our pizza?", "Your mobile app crashed multiple times during ordering."),
    (32, 32, "How did you like our pizza?", "The pesto drizzle was a game-changer. Loved it!"),
    (33, 33, "How did you like our pizza?", "I had to reheat the pizza as it wasn't warm enough."),
    (34, 34, "How did you like our pizza?", "The pepperoni slices were too thick and chewy."),
    (35, 35, "How did you like our pizza?", "Your meat lovers pizza is truly for meat lovers. So much variety!"),
    (36, 36, "How did you like our pizza?", "I'd suggest adding more green veggies as toppings."),
    (37, 37, "How did you like our pizza?", "The packaging was eco-friendly. Big thumbs up for that."),
    (38, 38, "How did you like our pizza?", "The onions were not fresh and had a weird aftertaste."),
    (39, 39, "How did you like our pizza?", "My order was missing the extra cheese I paid for."),
    (40, 40, "How did you like our pizza?", "The gluten-free crust was just as good as the regular one."),
    (41, 41, "How did you like our pizza?", "You forgot to send the chili flakes and oregano packets."),
    (42, 42, "How did you like our pizza?", "The caramelized onions and goat cheese combination was a hit!"),
    (43, 43, "How did you like our pizza?", "I received the wrong order but customer service sorted it out quickly."),
    (44, 44, "How did you like our pizza?", "A bit too greasy for my liking."),
    (45, 45, "How did you like our pizza?", "Loved the thin crust and fresh mozzarella."),
    (46, 46, "How did you like our pizza?", "The pizza was not sliced properly. Some slices were way too big."),
    (47, 47, "How did you like our pizza?", "The buffalo chicken topping was too spicy for me."),
    (48, 48, "How did you like our pizza?", "Quick delivery and polite delivery personnel."),
    (49, 49, "How did you like our pizza?", "I'd love if you introduced some stuffed crust options."),
    (50, 50, "How did you like our pizza?", "The Alfredo sauce base was a pleasant surprise. More of that, please!"),
    (51, 51, "How did you like our pizza?", "The pizza was undercooked and doughy in the middle."),
    (52, 52, "How did you like our pizza?", "I'd recommend offering combo deals with drinks and sides."),
    (53, 53, "How did you like our pizza?", "The pizza base was burnt and had a bitter taste."),
    (54, 54, "How did you like our pizza?", "The sausage and bell pepper combo is always my favorite."),
    (55, 55, "How did you like our pizza?", "The pizza was not hot and felt like it had been sitting out for a while."),
    (56, 56, "How did you like our pizza?", "You guys have the best white pizza in town."),
    (57, 57, "How did you like our pizza?", "I wish you had more seafood topping options."),
    (58, 58, "How did you like our pizza?", "The marinara sauce was too garlicky."),
    (59, 59, "How did you like our pizza?", "Keep up the great work! Always consistent with quality."),
    (60, 60, "How did you like our pizza?", "The extra toppings were not spread evenly. Most were in the center."),
    (61, 61, "How did you like our pizza?", "I'd like a crispier crust next time."),
    (62, 62, "How did you like our pizza?", "Your stuffed mushrooms side dish was fantastic!"),
    (63, 63, "How did you like our pizza?", "The pizza was overpriced for the size and toppings."),
    (64, 64, "How did you like our pizza?", "You need to improve on the freshness of toppings."),
    (65, 65, "How did you like our pizza?", "The anchovies were too salty."),
    (66, 66, "How did you like our pizza?", "I like the option to choose whole wheat crust."),
    (67, 67, "How did you like our pizza?", "The salad was wilted and not fresh."),
    (68, 68, "How did you like our pizza?", "Your seasonal specials are always something I look forward to."),
    (69, 69, "How did you like our pizza?", "The delivery fee is a bit too high."),
    (70, 70, "How did you like our pizza?", "Keep up the good work. I order from you guys every week!")
    ]
    return test_data

def main(operation, pipeline_job_id=0):
    print_debug("operation passed to main " + operation)
    training_data = get_training_data()
    print_debug("training data \n" + str(training_data))

    test_data = get_test_data()
    print_debug("test data \n" + str(test_data))
    prompt = generate_prompt(training_data)
    print_debug("prompt = \n" + prompt)
        
    if operation == "unit_test":
        
        labels = get_labels_for_feedbacks(test_data, prompt)
        clusters = cluster_labels(labels)
        output = generate_output(test_data, labels, clusters)
        for record in output:
            print(record)


if __name__ == "__main__":
    main("unit_test")