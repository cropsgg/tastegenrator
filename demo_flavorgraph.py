#!/usr/bin/env python3
"""
FlavorGraph Demo - Food Pairing Recommendations
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the graph data to get ingredient names
print("Loading FlavorGraph data...")
nodes_df = pd.read_csv("./input/nodes_191120.csv")
print(f"Loaded {len(nodes_df)} nodes from the graph")

# Create mapping from node_id to ingredient name
id_to_name = {}
name_to_id = {}
for _, row in nodes_df.iterrows():
    node_id = str(row['node_id'])
    name = row['name']
    node_type = row['node_type']
    is_hub = row['is_hub']
    
    id_to_name[node_id] = {
        'name': name,
        'type': node_type,
        'is_hub': is_hub
    }
    name_to_id[name] = node_id

# Load the embeddings
embedding_file = "./output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_2-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle"

print("Loading FlavorGraph embeddings...")
with open(embedding_file, "rb") as f:
    embeddings = pickle.load(f)

print(f"Loaded embeddings for {len(embeddings)} nodes")
print(f"Embedding dimension: {len(next(iter(embeddings.values())))}")

# Get some example ingredients
ingredients = []
compounds = []
for node_id, embedding in list(embeddings.items())[:100]:
    if node_id in id_to_name:
        info = id_to_name[node_id]
        if info['type'] == 'ingredient':
            ingredients.append((node_id, info['name'], info['is_hub']))
        else:
            compounds.append((node_id, info['name'], info['type']))

print(f"\nFound {len(ingredients)} ingredients and {len(compounds)} compounds in sample")

# Show some ingredients
print("\nSample ingredients from FlavorGraph:")
for i, (node_id, name, is_hub) in enumerate(ingredients[:10]):
    hub_status = "Hub" if is_hub == "hub" else "Regular"
    print(f"{i+1:2d}. {name:<30} ({hub_status})")

def find_similar_ingredients(target_name, top_k=5):
    """Find ingredients most similar to the target ingredient"""
    if target_name not in name_to_id:
        print(f"Ingredient '{target_name}' not found")
        return []
    
    target_id = name_to_id[target_name]
    if target_id not in embeddings:
        print(f"No embedding found for '{target_name}'")
        return []
    
    target_vector = embeddings[target_id].reshape(1, -1)
    similarities = []
    
    for node_id, embedding in embeddings.items():
        if node_id != target_id and node_id in id_to_name:
            info = id_to_name[node_id]
            if info['type'] == 'ingredient':  # Only compare with other ingredients
                sim = cosine_similarity(target_vector, embedding.reshape(1, -1))[0][0]
                similarities.append((info['name'], sim, info['is_hub']))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test with actual ingredients from the dataset
test_ingredients = [name for _, name, _ in ingredients[:5]]

print(f"\n{'='*70}")
print("FLAVORGRAPH FOOD PAIRING RECOMMENDATIONS")
print(f"{'='*70}")

for ingredient_name in test_ingredients:
    print(f"\nIngredients similar to '{ingredient_name}':")
    similar = find_similar_ingredients(ingredient_name, top_k=5)
    if similar:
        for i, (name, sim, is_hub) in enumerate(similar, 1):
            hub_status = " (Hub)" if is_hub == "hub" else ""
            print(f"  {i}. {name:<35} (similarity: {sim:.3f}){hub_status}")
    else:
        print("  No similar ingredients found")

# Summary statistics
ingredient_count = sum(1 for node_id in embeddings.keys() 
                      if node_id in id_to_name and id_to_name[node_id]['type'] == 'ingredient')
compound_count = len(embeddings) - ingredient_count

print(f"\n{'='*70}")
print("FLAVORGRAPH TRAINING SUMMARY")
print(f"{'='*70}")
print(f"✅ Total nodes embedded: {len(embeddings)}")
print(f"🥘 Food ingredients: {ingredient_count}")
print(f"🧪 Chemical compounds: {compound_count}")
print(f"📐 Embedding dimension: {len(next(iter(embeddings.values())))}")
print(f"🔄 Training iterations: 2")
print(f"🎯 Metapaths used: CHC + CHNHC + NHCHN")
print(f"💾 Output file: FlavorGraph+CSL-embedding_*.pickle")
print(f"{'='*70}")
print("✨ FlavorGraph successfully trained and ready for food pairing recommendations!")

