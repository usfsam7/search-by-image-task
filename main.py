import os
import glob
import sys
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load a pre-trained CNN model and remove the classification head
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  
model.eval()
model.to(device)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Auto-discover product images in images/products/
product_dir = os.path.join("images", "products")
patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
product_image_paths = []
for p in patterns:
    product_image_paths.extend(glob.glob(os.path.join(product_dir, p)))
product_image_paths = sorted(product_image_paths)

if not product_image_paths:
    sys.exit(f"No product images found in '{product_dir}'. Place images there or update the path.")

product_embeddings = []
valid_product_paths = []

for image_path in product_image_paths:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: could not open '{image_path}': {e}. Skipping.")
        continue

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        embedding = model(batch_t)  # shape: (1, 2048, 1, 1)
    emb = embedding.squeeze().cpu().numpy()  # shape: (2048,)
    product_embeddings.append(emb)
    valid_product_paths.append(image_path)

if len(product_embeddings) == 0:
    sys.exit(f"No valid product embeddings could be created from files in '{product_dir}'.")

product_embeddings = np.vstack(product_embeddings)  # shape: (N, 2048)

# Fit a k-NN model on the catalog embeddings
n_neighbors = min(5, len(product_embeddings))
knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
knn.fit(product_embeddings)


# Define a function to process a query image and find similar products
def find_similar_products(query_image_path, knn_model, product_paths, top_k=5):
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"Query image not found: {query_image_path}")

    query_img = Image.open(query_image_path).convert("RGB")
    query_img_t = preprocess(query_img)
    batch_t = torch.unsqueeze(query_img_t, 0).to(device)

    with torch.no_grad():
        query_embedding = model(batch_t)
    query_embedding = query_embedding.squeeze().cpu().numpy().reshape(1, -1)

    top_k = min(top_k, len(product_paths))
    distances, indices = knn_model.kneighbors(query_embedding, n_neighbors=top_k)

    similar_product_paths = [product_paths[i] for i in indices.flatten()]
    return similar_product_paths, distances


# Use the function
# Use the function
query_img_path = os.path.join("images", "search-image.png")
# try:
#     similar_paths, sim_scores = find_similar_products(query_img_path, knn, valid_product_paths)
#     print("Most similar products:", similar_paths)
#     print("Similarity scores:", sim_scores)
# except Exception as e:
#     print("Error:", e)

 #another form of the output
try:
    if not os.path.exists(query_img_path):
        raise FileNotFoundError(f"Query image not found: {query_img_path}")

    # Create query embedding
    query_img = Image.open(query_img_path).convert("RGB")
    query_img_t = preprocess(query_img)
    batch_t = torch.unsqueeze(query_img_t, 0).to(device)
    with torch.no_grad():
        query_embedding = model(batch_t)
    query_embedding = query_embedding.squeeze().cpu().numpy().reshape(1, -1)

    # Compute cosine similarities (range -1..1) and convert to percentage
    sims = cosine_similarity(query_embedding, product_embeddings).flatten()
    sims_pct = sims * 100.0

    # Print each product one-by-one with its similarity score
    print("\nSimilarity of each product:")
    for path, pct in zip(valid_product_paths, sims_pct):
        name = os.path.basename(path)
        print(f"{name}: {pct:.2f}%")

except Exception as e:
    print("Error:", e)
