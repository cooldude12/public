import argparse
from utils import print_debug, exec_sql


def main(question_id):
    sql_commands = [
        "DELETE FROM ai_pipeline.ai_split_text_llm WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_split_text_delimiter WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_split_text_delimiter WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_new_label_llm WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_label_vectordb WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_save_training_data WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_training_master WHERE question IN (SELECT SUBSTR(question, 1, 300) FROM triestai.questions WHERE question_id=%s);",
        "DELETE FROM ai_pipeline.ai_prep_data_llm WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_topic_llm WHERE question_id=%s;",
        "DELETE FROM ai_pipeline.ai_group_label_insight_llm WHERE question_id=%s;",
        "DELETE FROM ai.ai_pipeline_jobs WHERE question_id=%s;"
    ]

    for sql in sql_commands:
        exec_sql(sql, (question_id,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete records related to a specific question_id.')
    parser.add_argument('question_id', type=int, help='The question_id to delete records for')
    args = parser.parse_args()
    main(args.question_id)

"""

"""
