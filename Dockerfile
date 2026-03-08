# Use 3.12 (it's GA and stable now)
FROM public.ecr.aws/lambda/python:3.12

# 1. Install dependencies directly to the task root
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" --no-cache-dir

# 2. Copy the CONTENTS of your app folder into the task root
# This puts main.py, docs/, and rag_storage/ directly in /var/task/
COPY ./app/ ${LAMBDA_TASK_ROOT}/

# 3. Now your handler is just main.handler
CMD ["main.handler"]