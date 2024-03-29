FROM ghcr.io/planqk/job-template:latest-base-1.0.0

ENV ENTRY_POINT app.user_code.src.program:run

COPY requirements.txt ${USER_CODE_DIR}/
RUN pip install -r ${USER_CODE_DIR}/requirements.txt

COPY . ${USER_CODE_DIR}