import pandas as pd
import re
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

course_data = pd.read_csv('courses.csv')
course_data_2 = pd.read_csv('udemy_courses_2.csv')
course_data_2 = course_data_2[course_data_2['category'] == 'Development']
course_data_2 = course_data_2[['id', 'is_paid', 'num_subscribers', 'headline']]
course_final = pd.merge(course_data, course_data_2, on='id')
sample = course_final

# drop unneeded columns/features
sample = sample.drop(columns=['url', 'instructors_id', 'image', 'id', 'created', 'is_paid'])

# create a new technology column based on the course title
top_technologies = [
    "Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust", "Ruby", "Kotlin",
    "Swift", "PHP", "HTML", "CSS", "SQL", "NoSQL", "React", "Angular", "Vue.js", "Next.js",
    "Node.js", "Express.js", "Django", "Flask", "Spring Boot", ".NET Core", "FastAPI", "Laravel",
    "Ruby on Rails", "Firebase", "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Docker", "Kubernetes",
    "AWS", "Azure", "Google Cloud Platform (GCP)", "GraphQL", "REST APIs", "Git", "GitHub", "GitLab",
    "CI/CD", "Terraform", "Ansible", "Bash", "Linux", "Nginx", "Web Development", "Machine Learning",
    ".NET", "WordPress", "Data Science"
]

def extract_technologies(title):
    found = []
    matched_text = title.lower()

    for tech in top_technologies:
        escaped = re.escape(tech)
        pattern = rf'(^|[^a-zA-Z0-9]){escaped}($|[^a-zA-Z0-9])'

        if re.search(pattern, matched_text, re.IGNORECASE):
            # Avoid partial overlaps: don't match 'C' if 'C++' or 'C#' was matched
            if any(tech.lower() in f.lower() for f in found):
                continue
            found.append(tech)
    return ", ".join(found)

# Apply the function to create a new column
sample['technologies'] = sample['title'].apply(extract_technologies)
empty_rows_count = (sample['technologies'] == '').sum()

sample = sample[sample['technologies'].str.strip() != '']

empty_rows_count = (sample['technologies'] == '').sum()

# data preprocessing for duration_hours column
sample = sample.rename(columns={'duration': 'duration_hours'})
sample['last_update_date'] = pd.to_datetime(sample['last_update_date'], errors='coerce')

# Calculate days since last update from today's date
today = pd.to_datetime(datetime.today().date())
sample['days_since_last_update'] = (today - sample['last_update_date']).dt.days

def extract_numeric(value):
    numeric_value = ''.join(filter(str.isdigit, value))
    if numeric_value:
        return float(numeric_value)
    else:
        return None

# Apply the function to the 'duration' column
sample['duration_hours'] = sample['duration_hours'].apply(extract_numeric)
sample.to_csv('courses_final.csv', index=False)

sample_copy = sample.copy()

# number of subscribers vs technologies barplot
# Split multiple technologies into separate rows
sample_copy['technologies'] = sample_copy['technologies'].str.split(', ')
sample_copy = sample_copy.explode('technologies')

# Count technlogy by number of subscribers
top_techs = (
    sample_copy.groupby('technologies')['num_subscribers']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
top_techs_millions = top_techs / 1000000
ax = top_techs_millions.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.xlabel("Technologies")
plt.ylabel("Number of Subscribers (in millions)")
plt.title("Top 10 Technologies by Number of Subscribers")
plt.xticks(rotation=45)
plt.tight_layout()

# Add data labels on top of each bar
for i, value in enumerate(top_techs_millions):
    plt.text(i, value + 0.05, f'{value:.2f}M', ha='center', va='bottom')

plt.show()

# correlation heatmap
numerical_columns = sample_copy.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = sample_copy[numerical_columns].corr()
other_columns = [col for col in correlation_matrix.columns if col != 'num_subscribers']
ordered_columns = other_columns + ['num_subscribers']
ordered_matrix = correlation_matrix[ordered_columns].loc[ordered_columns]
plt.figure(figsize=(10, 8))
sns.heatmap(ordered_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()