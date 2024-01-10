from sentence_transformers import SentenceTransformer, util
import torch
import jellyfish

def compute_similarity(text1, text2):
    # Load the pre-trained model (this downloads the model on first run)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the texts to get their embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

    # Compute Damerau-Levenshtein distance
    damerau_levenshtein_distance = jellyfish.damerau_levenshtein_distance(text1, text2)
    max_possible_distance = max(len(text1), len(text2))
    normalized_damerau_levenshtein = 1 - (damerau_levenshtein_distance / max_possible_distance)

    return cosine_scores.item(), normalized_damerau_levenshtein
    
# Example usage
text1 = "This is a sample sentence."
text2 = "This is a different sample sentence."
cosine_similarity_score, damerau_levenshtein_distance = compute_similarity(text1, text2)
# Formatting the output to two decimal places
formatted_cosine_similarity_score = f"{cosine_similarity_score:.2f}"
formatted_normalized_damerau_levenshtein = f"{damerau_levenshtein_distance:.2f}"

print("text1=" + text1)
print("text2=" + text2)
print(f"Cosine Similarity score: {formatted_cosine_similarity_score}")
print(f"Damerau-Levenshtein distance: {formatted_normalized_damerau_levenshtein}")

