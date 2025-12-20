API connection to Gemini - Fetch dynamic cost estimation (in MXN and USD) based on input car model, year, damage part and severity. The estimation will also go through a verification with local database before finalized.


llm_estimator v2.py - Main Flask API server. Receives JSON → returns cost estimate using Gemini. 
list_models.py - Utility script to list available Gemini models your API key can access. 
cost_test.py -  Client script that sends a sample request to the API and prints the response. 
.env Stores your API keys and config (NOT included in GitHub). (GET YOUR OWN API KEY FROM GOOGLE AI Studio)


 Setup
 
Create and activate bash
python3 -m venv venv
source venv/bin/activate

Start the server:
python3 llm_estimator.py

You should see:
Running on http://127.0.0.1:8080

Open Second terminal:
Run Cost_test.py

source venv/bin/activate
python3 cost_test.py

You will see output:

  "label": "rear-bumper-dent",
  "severity": 0.70,
  "estimate": {
    "parts_cost": 5200,
    "labor_cost": 2750,
    "total_cost": 7950


    


  
