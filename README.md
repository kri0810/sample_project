# Offensive Text Detection On Social Media

This project in a Natural Language Processing(NLP) application on determining whether a social media tweet is offensive or not. If its offensive then determining whether it is targeted offense or untargeted offensive. If it is targeted offense then determining whether it is group targeted, individual targeted or others.

## setup

 ### Step 1: Create a virtual environment to isolate our packages and dependencies
python -m venv venv
Activate the virtual env
On windows use .\venv\Scripts\activate


 ### Step 2: Clone the Github Repository
Navigate to the directory where you want to clone the repository
cd /path/to/your/directory

Clone the repository
git clone https://github.com/username/repository.git


### Step 3: Add all the requirements mentioned in requirements.txt
pip install -r requirements.txt


### Step 4: Files description
requirements.txt: Contains all necessary dependencies for the project.
Inside folder named final:
utils.py: Contains necessary functions used in the codebase.
backend.py: FastAPI code for creating endpoints.
frontend.py: Streamlit code for creating the user interface.
main.py: Integrates frontend.py and backend.py

Model files:

random_forest_model_a.pkl: Random Forest model for predicting whether the text is offensive or not offensive.

If offensive then 2nd model will get triggered

random_forest_model_b.pkl: Random Forest model for predicting whether the text is Targeted offense or Untargeted offense.

If targeted then 3rd model will get triggered

random_forest_model_c.pkl: Random Forest model for predicting whether the text is Individual Targeted Offense or whether it is Group Targeted Offense or whether it belongs to Others. 

Vectorization file:
tfidf_vectorizer.pkl:this project uses tfidf vectorization


### Step 5:Running the project
Instructions for running the backend and frontend components.

Frontend
streamlit run frontend.py

Backend
uvicorn backend:app --reload --port 8001

Integrated Module (main)
streamlit run main.py


For questions or issues, please contact us on (mehtakrisha0810@gmail.com)
