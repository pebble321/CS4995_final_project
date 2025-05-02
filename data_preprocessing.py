import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('courses_final.csv')
# drop unneeded column
df = df.drop(columns=['last_update_date'])

# transform textual features into word embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')
# 🔁 Combine 'title' and 'headline' into one column for embedding
df['textual_embeddings'] = df['title'].fillna('') + ' ' + df['headline'].fillna('')

# ⚙️ Encode the combined text into embeddings
embeddings = model.encode(df['textual_embeddings'].tolist(), show_progress_bar=True)

# Optionally, convert embeddings to a DataFrame (optional but helpful for inspection)
embedding_df = pd.DataFrame(embeddings)
df_embeddings = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)


# scale the original embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# apply PCA with 95% variance to reduce dimension
pca = PCA(n_components=0.95, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_scaled)
df_embeddings_scaled = pd.DataFrame(embeddings_pca)

# transform categorical feature(technlogies) into binary representation
top_technologies = [
    "Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "Ruby", "Kotlin",
    "Swift", "PHP", "HTML", "CSS", "SQL", "NoSQL", "React", "Angular", "Vue.js", "Next.js",
    "Node.js", "Express.js", "Django", "Flask", "Spring Boot", ".NET Core", "FastAPI", "Laravel",
    "Ruby on Rails", "Firebase", "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Docker", "Kubernetes",
    "AWS", "Azure", "Google Cloud Platform (GCP)", "GraphQL", "REST APIs", "Git", "GitHub", "GitLab",
    "CI/CD", "Terraform", "Ansible", "Bash", "Linux", "Nginx", "Web Development", "Machine Learning",
    ".NET", "WordPress", "Data Science"
]

def multi_hot_encode(technologies, all_technologies):
    encoded_vector = np.zeros(len(all_technologies))

    for tech in technologies:
        if tech in all_technologies:
            encoded_vector[all_technologies.index(tech)] = 1
    return encoded_vector

encoded_technologies = df['technologies'].apply(lambda x: multi_hot_encode(x, top_technologies))
encoded_technologies_df = pd.DataFrame(encoded_technologies.tolist(), columns=top_technologies)
df_embeddings_scaled = pd.concat([df_embeddings_scaled, encoded_technologies_df], axis=1)

# scale all other numerical features
numerical_columns = ['rating', 'num_reviews', 'num_published_lectures', 'duration_hours', 'days_since_last_update', 'num_subscribers']
numerical_data = df[numerical_columns]

# apply standard scaling to numerical features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_data)
numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_columns)
df_embeddings_scaled = pd.concat([df_embeddings_scaled, numerical_scaled_df], axis=1)

# save the preprocessed version and it will be used for model training
df_embeddings_scaled.to_csv('embeddings_scaled_final.csv', index=False)
print("File saved as 'embeddings_scaled_final.csv'")