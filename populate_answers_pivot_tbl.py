import os
import sys
import csv
import time    
from utils import print_debug, exec_sql, display_output
LOAD_TBL = True
JOB_INTERVAL_IN_HOURS=1

def pivot_data(template_publish_id):
    sql = f"""
        SELECT d.question_id, d.question, e.user_id, e.id as answer_id, ans_txt
        FROM triestai.template_publish_details a1
        JOIN triestai.forms a ON a1.form_id=a.form_id
        JOIN triestai.factors b ON a.root_factor_id=b.parent_id
        JOIN triestai.questions d on d.factor_id=b.factor_id
        JOIN triestai.answers e on d.question_id=e.question_id
        WHERE a1.template_publish_id={template_publish_id}
        ORDER BY e.user_id, d.question_id
    """
    sql = f"""
        SELECT 
            d.question_id, 
            d.question, 
            e.user_id, 
            e.id as answer_id, 
            COALESCE(qo.ans_txt, e.ans_txt) AS ans_txt
        FROM triestai.template_publish_details a1
        JOIN  triestai.forms a ON a1.form_id = a.form_id
        JOIN triestai.factors b ON a.root_factor_id = b.parent_id
        JOIN triestai.questions d ON d.factor_id = b.factor_id
        JOIN triestai.answers e ON d.question_id = e.question_id
        LEFT JOIN (
            SELECT 
                b.answer_id, 
                GROUP_CONCAT(CONCAT(a.choice_word, ': ', b.ans_ch_text) SEPARATOR '\n') AS ans_txt
            FROM triestai.que_options a 
            JOIN triestai.answers_options b ON a.choice_id = b.choice_id
            GROUP BY b.answer_id
        ) qo ON e.id = qo.answer_id
        WHERE a1.template_publish_id = {template_publish_id} 
        ORDER BY e.user_id, d.question_id
    """

    success, result = exec_sql(sql)
    print_debug("number of answer records in stidy = " + str(len(result)))
    if not success:
        print(f"Failed to execute query: {result}")
        return

    # Pivot the data
    pivot_table = {}
    question_mapping = {}
    question_id_mapping = {}
    for row in result:
        question_id, question, user_id, _, answer = row
        question_id_mapping[question] = question_id

        if user_id not in pivot_table:
            pivot_table[user_id] = {}

        pivot_table[user_id][question] = answer

    if LOAD_TBL:
        print_debug("---------------------------------------------------")
        print_debug("now loading data in tbl for template_publish_id = " + str(template_publish_id))
        column_names = get_column_names_from_pivot_data(pivot_table, question_id_mapping)
        question_mapping = build_question_mapping(column_names)

        if not populate_static_table(template_publish_id, pivot_table, question_mapping, question_id_mapping):
            print_debug("Failed to populate the static table")
        print_debug("loading data in tbl done")
    else:
        print_debug("Load tbl skipped")

def build_question_mapping1(column_names, max_questions=12):
    question_mapping = {}
    #print_debug("number of questions = " + str(len(question_text)))
    for i, question_text in enumerate(column_names[1:], 1):  # Skip 'user_id' and start from 1
        if i > max_questions:
            print_debug("questions count > 12, taking first 12 questions")
            break  # Stop mapping if more than max_questions
        question_mapping[question_text] = f'q{i}'

    return question_mapping

def build_question_mapping(column_names, max_questions=12):
    question_mapping = {}
    num_questions = len(column_names) - 1  # Subtracting 1 for 'user_id' column
    print_debug(f"Total number of questions: {num_questions}")

    for i, question_text in enumerate(column_names[1:], 1):  # Skip 'user_id' and start from 1
        if i > max_questions:
            print_debug("Question count > 12, taking first 12 questions")
            break  # Stop mapping if more than max_questions
        question_mapping[question_text] = f'q{i}'

    return question_mapping

def populate_static_table(template_publish_id, pivot_data, question_mapping, question_id_mapping, num_questions=12):
    column_names = ['template_publish_id', 'data_type_category', 'user_id'] + [f'q{i}' for i in range(1, num_questions + 1)]

    sql_delete = f"DELETE FROM triestai.answers_pivot WHERE template_publish_id={template_publish_id}"
    exec_sql(sql_delete)

    # Insert question texts and question IDs first
    insert_questions_row(template_publish_id, question_mapping, column_names, question_id_mapping)

    records_inserted = 0  # Initialize counter for inserted records
    # Insert answer texts
    for user_id, answers in pivot_data.items():
        row_data = [str(template_publish_id), '"answer_text"', f'"{user_id}"']
        for i in range(1, num_questions + 1):
            q_key = f'q{i}'
            question_text = [key for key, value in question_mapping.items() if value == q_key]
            answer = answers.get(question_text[0], 'NULL') if question_text else 'NULL'
            row_data.append(f'"{answer}"' if answer != 'NULL' else 'NULL')

        row_sql = ', '.join(row_data)
        insert_row_sql = f'INSERT INTO triestai.answers_pivot ({", ".join(column_names)}) VALUES ({row_sql});'
        success, _ = exec_sql(insert_row_sql)
        if success:
            records_inserted += 1  # Increment counter if insertion is successful

    print_debug(f"Total records inserted: {records_inserted}")
    return True


def insert_questions_row(template_publish_id, question_mapping, column_names, question_id_mapping):
    row_data_text = [str(template_publish_id), '"question_text"', '"N/A"']
    row_data_id = [str(template_publish_id), '"question_id"', '"N/A"']

    for i in range(1, min(len(column_names) - 3, 12) + 1):
        q_key = f'q{i}'
        question_text = next((key for key, value in question_mapping.items() if value == q_key), 'NULL')
        question_id = question_id_mapping.get(question_text, 'NULL')

        row_data_text.append(f'"{question_text}"' if question_text != 'NULL' else 'NULL')
        row_data_id.append(f'"{question_id}"' if question_id != 'NULL' else 'NULL')

    # Insert question text row
    row_sql = ', '.join(row_data_text)
    insert_row_sql = f'INSERT INTO triestai.answers_pivot ({", ".join(column_names)}) VALUES ({row_sql});'
    exec_sql(insert_row_sql)

    # Insert question ID row
    row_sql = ', '.join(row_data_id)
    insert_row_sql = f'INSERT INTO triestai.answers_pivot ({", ".join(column_names)}) VALUES ({row_sql});'
    exec_sql(insert_row_sql)

def get_column_names_from_pivot_data(pivot_data, question_id_mapping):
    question_texts = list(question_id_mapping.keys())
    column_names = ['user_id'] + question_texts
    return column_names


def get_template_publish_ids():
    sql = f"""
    SELECT DISTINCT template_publish_id FROM (
        SELECT a1.template_publish_id
        FROM triestai.template_publish_details a1
        JOIN triestai.forms a ON a1.form_id=a.form_id
        JOIN triestai.factors b ON a.root_factor_id=b.parent_id
        JOIN triestai.questions d ON d.factor_id=b.factor_id
        JOIN triestai.answers e ON d.question_id=e.question_id
        WHERE e.record_updated_ts > NOW() - INTERVAL {JOB_INTERVAL_IN_HOURS} HOUR
    ) a;
    """
    success, result = exec_sql(sql)
    if success:
        return [row[0] for row in result]
    else:
        print(f"Failed to fetch template_publish_ids: {result}")
        return []

def main(category, template_publish_id=None):
    if category == "unit_test":
        if template_publish_id is None:
            print("Error: template_publish_id is required for unit_test.")
            return
        pivot_data(template_publish_id)

    elif category == "db":
        template_publish_ids = get_template_publish_ids()
        for t_id in template_publish_ids:
            pivot_data(t_id)
    else:
        print("Invalid category. Please use 'unit_test' or 'db'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_answers.py <category> [template_publish_id]")
        sys.exit(1)

    category = sys.argv[1]
    template_publish_id = sys.argv[2] if len(sys.argv) > 2 else None

    main(category, template_publish_id)

"""
if __name__ == "__main__":
    LOAD_DATA = False
    if len(sys.argv) < 2:
        print("Usage: python get_answers.py <template_publish_id> ")
    else:
        template_publish_id = sys.argv[1]
        pivot_data(template_publish_id)

"""