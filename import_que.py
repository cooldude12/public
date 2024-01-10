import pymysql
import os

def exec_sql(sql, give_id=False):
    connection = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PWD'),
        database=os.getenv('DB_NAME')
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            insert_id = cursor.lastrowid
            connection.commit()
            result = cursor.fetchall()
            if give_id == True:
                return insert_id
            else:
                return result
    except Exception as e:
        print(sql)
        print(f"\nError executing SQL: {e}")
        return str(e)
        sys.exit(1)
    finally:
        connection.close()

existing_question_id = input("Enter the existing Question ID:")
# target_study_id = input("Enter the target Study ID:")
target_study_id = 5435

sql = f"SELECT * FROM triestai.template_publish_details WHERE TEMPLATE_PUBLISH_ID = {target_study_id};"
tgt_study = exec_sql(sql)

sql = f"SELECT * FROM triestai.forms WHERE FORM_ID = {tgt_study[0][6]};"
tgt_form = exec_sql(sql)

sql = f"SELECT * FROM triestai.factors WHERE parent_id = {tgt_form[0][4]};"
tgt_factor = exec_sql(sql)

sql = f"SELECT * FROM triestai.questions WHERE factor_id = {tgt_factor[0][0]};"
tmp_ques = exec_sql(sql)

sql = f"""INSERT INTO triestai.questions (QUESTION,DESCRIPTION,QUE_TYPE_ID,MANDATORY,QUE_SEQUENCE,
    FACTOR_ID,NUM_OPTIONS,READ_ONLY,HAS_PICTURE,HAS_TEXT,NUM_ONLY) SELECT QUESTION,DESCRIPTION,QUE_TYPE_ID,MANDATORY,{len(tmp_ques)+1},{tgt_factor[0][0]},
    NUM_OPTIONS,READ_ONLY,HAS_PICTURE,HAS_TEXT,NUM_ONLY FROM triestai.questions WHERE question_id = {existing_question_id};"""
question_id = exec_sql(sql, True)

sql = f"select * from triestai.tasks where TEMPLATE_PUBLISH_ID = {target_study_id};"
tasks = exec_sql(sql)

sql = f"SELECT * FROM triestai.answers WHERE question_id = {existing_question_id} LIMIT {len(tasks)};"
existin_ans = exec_sql(sql)

for idx, ans in enumerate(existin_ans):
    sql = f"""INSERT INTO `triestai`.`answers`
            (`USER_ID`,`TASK_ID`,`FORM_ID`,`QUESTION_ID`,`ANS_TXT`,`ANS_CHOICE_ID`,`DATE_MODIFIED`,`ANS_MODE`)
            VALUES
            ({tasks[idx][4]},{tasks[idx][0]},{tgt_study[0][6]},{question_id},"{ans[5]}","{ans[6]}",now(),4);"""
    new_ans = exec_sql(sql, True)

print(f"Inserted {len(existin_ans)} answers new Question ID {question_id}")

