### Use a python virtual environment
Example: `source fastapirashi/bin/activate`

### Install from requirements.txt file
Example: `pip3 install -r requirements.txt`

#### Be sure to add all installs to requirements.txt file
Example: `pip3 freeze > requirements.txt`

### Run in dev mode (so changes refresh)
Command: `uvicorn main:app --reload`