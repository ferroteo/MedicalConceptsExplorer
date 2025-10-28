# MedicalConceptsExplorer
Local app for the exploration of the closest medical concept given a specific text input. 

**Instructions:**<br>
download all the files in a directory on your computer, from the directory you now just need to:

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install required libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Initialize required components locally.<br>

pre-calculate embeddings (GatorTron model as default)
```bash
python PreComputeEmbeddings.py
```
Otherwise pre-calculate embeddings from multiple or all available models, like this:
```bash
python PreComputeEmbeddings.py --model all
python PreComputeEmbeddings.py --model bridge gatortron
```
Note: if you have it, nmove and unzip the BRIDGE model (in ./models folder)

4. Start the application:
```bash
streamlit run WebApp.py
```

# Versions  

## Version 1
- Embedding model: GatorTron, BRIDGE
- Objective: Look for closest ICD-10-CM concept in the model embedding space