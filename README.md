# MedicalConceptsExplorer
Local app for the exploration of the closest medical concept given a specific text input. 

**Instructions:**<br>
download all the files in a directory on your computer, from this directory you now just need to:
- Initialize all required components by runnning: _bash Initialize.sh_ <br> 
(sets up a python virtual env and pre computes embeddings locally)
- Start exploring by running: _streamlit run WebApp.py_ <br>
(make sure virtual env is active before running, run source venv/bin/activate otherwise)

## Version 1
- Embedding model: GatorTron, BRIDGE
- Objective: Look for closest ICD-10-CM concept in the model embedding space