from transformers import BertTokenizer, BertModel
import torch
import re
import os
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
# First, we read all the .c files and aggregate their contents into the 'code' variable

code = ""
folder_path = "Electric-Water-Heater-main"

for subdir, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.c'):
            with open(os.path.join(subdir, file), 'r', encoding="utf-8", errors='ignore') as f:
                code += f.read() + "\n\n"

# Filter out less significant functions and variables

# For simplicity, let's define a list of keywords that might suggest a function or variable is of lower significance
insignificant_keywords = ["init", "update", "set", "get", "clear", "enable", "disable", "start", "stop", "read", "write"]

def is_significant(name):
    for keyword in insignificant_keywords:
        if keyword in name.lower():
            return False
    return True

# Step 1: Identify Features
function_names = re.findall(r"(void|float|int|double|bool) (\w+)\(", code)
global_vars = [match[1] for match in re.findall(r"(float|int|double|bool) (\w+) =", code) if "{" not in code.split(match[0])[0]]

# Filter out less significant functions and variables
significant_function_names = [name[1] for name in function_names if is_significant(name[1])]
significant_global_vars = [var for var in global_vars if is_significant(var)]

features = significant_function_names + significant_global_vars

# Step 2: Identify Functional Requirements
conditions = re.findall(r"(if|while|for) \(([^)]+)\)", code)
functional_requirements = [cond[1] for cond in conditions]

# Step 3: Identify Non-Functional Requirements
comments = re.findall(r"// (.+)", code)
non_functional_requirements = comments

# Step 4: Map Requirements to Features using BERT

all_requirements = functional_requirements + non_functional_requirements

feature_embeddings = {}
requirement_embeddings = {}

for feature in features:
    inputs = tokenizer(feature, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    feature_embeddings[feature] = output.last_hidden_state.mean(dim=1).squeeze().numpy()

for requirement in all_requirements:
    inputs = tokenizer(requirement, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    requirement_embeddings[requirement] = output.last_hidden_state.mean(dim=1).squeeze().numpy()

feature_to_requirements = {feature: [] for feature in features}
for requirement, req_embedding in requirement_embeddings.items():
    similarities = {}
    for feature, feat_embedding in feature_embeddings.items():
        similarity = cosine_similarity([req_embedding], [feat_embedding])[0][0]
        similarities[feature] = similarity

    best_matching_feature = max(similarities, key=similarities.get)
    feature_to_requirements[best_matching_feature].append(requirement)

len_features = len(features)
len_functional_requirements = len(functional_requirements)
len_non_functional_requirements = len(non_functional_requirements)

fig, ax = plt.subplots(2, 1, figsize=(14, 12))

# Electric Water Heater
ax[0].bar(features_waterheater, coherence_bow_waterheater, width=0.4, label="BoW", align="center", color="green")
ax[0].bar(features_waterheater, coherence_tfidf_waterheater, width=0.4, label="TF-IDF", align="edge", color="c")
ax[0].set_ylabel("Coherence Score", fontsize=12)
ax[0].set_title("Electric Water Heater")
ax[0].legend()
ax[0].tick_params(axis="x", rotation=90)
ax[0].grid(True)

# Microwave System
ax[1].bar(features_microwave, coherence_bow_microwave, width=0.4, label="BoW", align="center", color="green")
ax[1].bar(features_microwave, coherence_tfidf_microwave, width=0.4, label="TF-IDF", align="edge", color="c")
ax[1].set_ylabel("Coherence Score", fontsize=10)
ax[1].set_title("Microwave System")
ax[1].legend()
ax[1].tick_params(axis="x", rotation=90)
ax[1].grid(True)
plt.tight_layout()
plt.show()