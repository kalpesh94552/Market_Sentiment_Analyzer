
# Market Sentiment Analyzer (Assignment Submission)

This repository is an assignment for analyzing market sentiment for a given company using Azure OpenAI, LangChain, and MLflow.

---

## Setup Instructions

1. **Clone the Repository**
	```bash
	git clone <your-repo-url>
	cd Market_Sentiment_Analyzer
	```

2. **Create and Activate Virtual Environment**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. **Install Requirements**
	```bash
	pip install -r requirement.txt
	```

4. **Configure Environment Variables**
	- Edit the `.env` file and fill in your Azure OpenAI and MLflow credentials.
	- Example `.env`:
	  ```
	  AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.api.cognitive.microsoft.com/"
	  AZURE_OPENAI_API_KEY="<your-azure-openai-key>"
	  AZURE_OPENAI_API_VERSION="2024-08-01-preview"
	  AZURE_OPENAI_DEPLOYMENT="gpt4o"
	  ```

---


## Azure and MLflow API Configuration

### Azure AI Foundry (Deploying GPT-4o)

1. **Create an Azure AI Foundry Resource**
	- Sign in to the [Azure Portal](https://portal.azure.com/).
	- Click **Create a resource** and search for **Azure OpenAI** (AI Foundry Models).
	- Click **Create** and fill in:
	  - **Subscription:** Your Azure subscription.
	  - **Resource group:** Create or select an existing group.
	  - **Region:** Choose a region (closest to you for lower latency).
	  - **Name:** A descriptive name for your resource.
	  - **Pricing Tier:** Standard (default).
	- Click **Next** to configure network security (default: allow all networks for development).
	- Click **Review + create** and then **Create**.

2. **Deploy the GPT-4o Model**
	- Go to your new Azure OpenAI (AI Foundry) resource in the portal.
	- Under **Management**, select **Deployments**.
	- Click **Create new deployment**.
	  - **Select a model:** Choose `gpt-4o` from the dropdown.
	  - **Deployment name:** Choose a name (e.g., `gpt4o`).
	- Click **Create** and wait for deployment to complete.

3. **Get API Details**
	- In your Azure OpenAI resource, go to **Keys and Endpoint**.
	- Copy the **Endpoint** and **API Key**.
	- Use the **API Version** shown in the portal (e.g., `2024-08-01-preview`).
	- Use your deployment name from the previous step.

4. **Update Your `.env` File**
	```
	AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.api.cognitive.microsoft.com/"
	AZURE_OPENAI_API_KEY="<your-azure-openai-key>"
	AZURE_OPENAI_API_VERSION="2024-08-01-preview"
	AZURE_OPENAI_DEPLOYMENT="gpt4o"
	```

### MLflow

- MLflow runs locally by default.  
- To view experiment results, start the MLflow UI:
  ```bash
  mlflow ui
  ```
- Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## How to Run

### From Jupyter Notebook

1. Open `assignment.ipynb` in VS Code or Jupyter.
2. Run all cells in order.
3. The main function call is:
	```python
	run_sentiment_pipeline("Microsoft")
	```

### From Terminal

You can also run the pipeline from the terminal:
```bash
python sentiment.py --company "Microsoft"
```

---

## Notes

- Ensure your `.venv` is activated before running any commands.
- The examiner should only need to update the `.env` file with their own Azure credentials.
- All experiment runs and results are tracked in MLflow and can be viewed in the MLflow UI.

---

**This is an assignment submission. Please follow the above steps to evaluate the code.**