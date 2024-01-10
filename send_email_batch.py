import boto3
import smtplib
import utils
from utils import exec_sql, print_debug, display_output
import sys
# import necessary modules
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re 

# Fetch the email information from the tables
status, result = exec_sql("SELECT id, email_from_address, email_subject, email_body \
                          FROM ai_pipeline.email_job WHERE job_status = 'scheduled' \
                          order by id asc LIMIT 1")

print_debug("results count = " + str( len(result)))

if not status:
    print_debug(f"Failed to execute SQL: {result}")
    exit()
# New code begins
elif len(result) < 1: # it alwasy returns the heade row, hence <= 1
    print_debug("No scheduled jobs found.")
    exit()
elif len(result) == 1:
    print_debug(str(result[0]))
else:
    print_debug("nothing")

# Process the fetched results
email_job_id, email_from_address, email_subject, email_content = result[0]
print_debug("job id " + str(email_job_id) + "\n subject:" + email_subject + " from:" + email_from_address)

sql_query = f"SELECT email_recipient FROM ai_pipeline.email_recipients where email_job_id={email_job_id}"
print_debug(sql_query)
status, result = exec_sql(sql_query)
if not status:
    print_debug(f"Failed to execute SQL: {result}")
    exit()

elif len(result) < 1: # it alwasy returns the heade row, hence <= 1
    print_debug("No recipient email address found for this email task. closing the job")
    # close the job 
    update_sql = f"UPDATE ai_pipeline.email_job SET job_status = 'sent' WHERE id = {email_job_id}"
    exec_sql(update_sql)
    exit()

# process begins
# Update the job status in the email_content table
print_debug("now kicking off the batch job")
update_sql = f"UPDATE ai_pipeline.email_job SET job_status = 'inprogress' WHERE id = {email_job_id}"
status, result_update = exec_sql(update_sql)

recipients = []
for row in result:
    email_address = row[0].strip() # strip() is used to remove leading/trailing whitespaces
    if '@' in email_address: # Ensure the email address is valid
        recipients.append(email_address)

print_debug("Recipeinet list \n" + str(recipients))

# Configure the SMTP client with AWS SES credentials
ses_client = boto3.client('ses', region_name='ap-south-1')
smtp_client = smtplib.SMTP('email-smtp.ap-south-1.amazonaws.com', port=587)
smtp_client.starttls()
smtp_client.login('AKIAUC6UBX3US3GQ2665', 'BK0ne500WTZABsgNFQleEU44es5dkIP68PjdMjQHUXgm')


# Compose the email
subject = email_subject
# Replace newline and carriage return characters
email_content = re.sub(r'\s+', ' ', email_content)
email_content = email_content.replace('\\n', '').replace('\\r', '')
email_content = email_content.replace('\n', '').replace('\r', '')

body = email_content.encode('utf-8')  # Encode the email content using UTF-8
from_address = email_from_address

to_addresses = [recipient for recipient in recipients]

print_debug("recipients list ")
display_output(to_addresses)

# create MIMEMultipart message
msg = MIMEMultipart('alternative')

# Add subject, from and to lines
msg['Subject'] = subject
msg['From'] = from_address
#msg['To'] = ', '.join(to_addresses)

# Add HTML content
#html_part = MIMEText(body, 'html')  # <-- this line ensures your email is treated as HTML
html_part = MIMEText(body.decode('utf-8'), 'html') 
print_debug(" subject:" + email_subject + " body:" + email_content)
msg.attach(html_part)

# Send the email one at a time
for recipient in recipients:
  print_debug("sending email to " + recipient)
  del msg['To']
  msg['To'] = recipient
  smtp_client.sendmail(from_address, recipient, msg.as_string())  # <-- use msg.as_string()

# Update the job status in the email_content table
update_sql = f"UPDATE ai_pipeline.email_job SET job_status = 'sent' WHERE id = {email_job_id}"
status, result = exec_sql(update_sql)
smtp_client.quit()

"""
TEST DATA for testing. 

need to populate email_job and email reciipients table. here are sample entries of 2 email bodies
and recipients list. 

after inserting you can test from cli : python3 send_email_batch.py

truncate table email_job;

insert into email_job (id,email_from_address,email_subject,email_body, job_status ) values (1,'ben.goswami@triestai.com','test subject', 'test body \n test body \n link: www.amazon.com','scheduled');
insert into email_job (id,email_from_address,email_subject,email_body, job_status ) values (2,'ben.goswami@triestai.com','test subject for harlem foods', 'test body foods \n test body \n link: www.amazon.com','scheduled');
INSERT INTO email_job (id, email_from_address, email_subject, email_body, job_status)
VALUES (3, 'ben.goswami@gmail.com', 'test subject for html foods','
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Simple HTML Email</title>
</head>
<body>
  <h1>Welcome to my HTML Email</h1>
  <p>
    This is a sample HTML email with some basic elements:
  </p>
  <ol>
    <li>
      <a href="https://example.com">HTML Link</a>
    </li>
    <li>
      Unicode character: &copy;
    </li>
    <li>
      Footer: This is the footer of the email.
    </li>
  </ol>
</body>
</html>', 
'scheduled');
INSERT INTO email_job (id, email_from_address, email_subject, email_body, job_status)
VALUES (3, 'ben.goswami@gmail.com', 'test subject for xyz foods',
        CAST('Hi,

        Luke Lobo has requested your inputs on "luke demo 334"

        To fill your inputs, please click the link – luke demo 334
        Once you have filled-in your inputs, click on the COMPLETE button to submit.

        You can also login to triestai.com and under Pending Study access "luke demo 334".


        Best regards,
        TriestAI Support
        (Please, DO NOT REPLY to this auto-generated mail.)
        If you do not wish to receive mail notifications click to unsubscribe.
        copyright: 2023 TriestAI, Inc. | support@triestai.com'
        AS BINARY), 'scheduled');


commit; 
truncate table email_recipients;
insert into email_recipients (email_job_id,email_recipient) values (1,'ben.goswami@gmail.com');
insert into email_recipients (email_job_id,email_recipient) values (1,'ben.goswami@triestai.com');
insert into email_recipients (email_job_id,email_recipient) values (2,'ben.goswami@gmail.com');
insert into email_recipients (email_job_id,email_recipient) values (2,'ben.goswami@triestai.com');
insert into email_recipients (email_job_id,email_recipient) values (3,'ben.goswami@gmail.com');
insert into email_recipients (email_job_id,email_recipient) values (3,'ben.goswami@triestai.com');
commit; 

"""