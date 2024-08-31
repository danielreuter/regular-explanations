# regular-explanations

This repo contains exploratory code for a mechanistic interpretability technique that leverages the syntactic structure of code to generate programmatic explanations of SAE features that can be cheaply verified across the full distribution. 

- `explore.ipynb` contains the first pass. 
- `create_activation_data.py` is a script that generates the activation dataset
- `simulate.ipynb` uses the activation data to test how well we can simulate SAE features using simple regexes. 
