#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import os

zip_path = 'athlete_events.csv.zip'
extract_to = 'unzipped_athlete_data'

# Create folder and extract
os.makedirs(extract_to, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Zip file extracted!")


# In[2]:


extracted_files = os.listdir(extract_to)
print("Extracted files:", extracted_files)


# In[3]:


import pandas as pd

csv_path = os.path.join(extract_to, 'athlete_events.csv')
df_athlete = pd.read_csv(csv_path)

# Show first few rows
df_athlete.head()



# In[4]:


print("Columns:", df_athlete.columns.tolist())
df_athlete.describe()


# In[5]:


df_noc = pd.read_csv("noc_regions.csv")
df_noc.head()
df_noc.describe()


# In[6]:


get_ipython().system('pip install pandasql')


# In[7]:


from pandasql import sqldf
pysql = lambda q: sqldf(q, globals())


# In[8]:


pysql("SELECT *  From df_athlete limit 10")


# In[9]:


pysql("Select * from df_noc")


# In[10]:


pysql("SELECT Team, COUNT(Medal) AS Total_Medals_in_120_Y FROM df_athlete WHERE Medal IS NOT NULL GROUP BY Team ORDER BY Total_Medals_in_120_Y DESC")


# In[11]:


query = """
SELECT n.region AS Team, COUNT(a.Medal) AS Total_Medals_in_120_Y
FROM df_athlete a
JOIN df_noc n 
    ON a.NOC = n.NOC
WHERE a.Medal IS NOT NULL
GROUP BY n.region
ORDER BY Total_Medals_in_120_Y DESC
"""
medals_by_team = pysql(query)
medals_by_team.head(15)


# In[12]:


#ratio of number of male and female athletes by year

import matplotlib.pyplot as plt

query = """
SELECT Year, Sex, COUNT(DISTINCT ID) AS Athlete_Count
FROM df_athlete
GROUP BY Year, Sex
ORDER BY Year, Sex
"""
result = pysql(query)

pivot_df_athlete = result.pivot(index='Year', columns='Sex', values='Athlete_Count').fillna(0)
pivot_df_athlete.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Number of Male and Female Athletes by Year')
plt.xlabel('Olympic Year')
plt.ylabel('Number of Athletes')
plt.legend(title='Sex')
plt.show()


# In[13]:


#age distribution of athletes

query = """
SELECT 
    CASE
        WHEN Age < 18 THEN '<18'
        WHEN Age BETWEEN 18 AND 30 THEN '18-30'
        WHEN Age BETWEEN 31 AND 40 THEN '31-40'
        WHEN Age BETWEEN 41 AND 50 THEN '41-50'
        WHEN Age BETWEEN 51 AND 55 THEN '51-55'
        WHEN Age > 55 THEN '>55'
        ELSE 'Unknown'
    END AS Age_Group,
    COUNT(*) AS Count
FROM df_athlete
WHERE Age IS NOT NULL
GROUP BY Age_Group
ORDER BY Count DESC
"""

# Run the query
age_distribution = pysql(query)

# Plot pie chart
plt.figure(figsize=(5, 5))
plt.pie(age_distribution['Count'], labels=age_distribution['Age_Group'])
plt.title('Distribution of Athlete Ages')
plt.axis('equal')
plt.legend(age_distribution['Age_Group'], title='Age Group', loc='center left', bbox_to_anchor=(1.55, 0.5))
plt.show()


# In[14]:


#Age in each sport max and min

query = """
SELECT Sport,
       ROUND(AVG(Age), 1) AS Avg_Age,
       MIN(Age) AS Min_Age,
       MAX(Age) AS Max_Age,
       COUNT(*) AS Medalists_Count
FROM df_athlete
WHERE Medal IS NOT NULL AND Age IS NOT NULL
GROUP BY Sport
ORDER BY Medalists_Count DESC
"""

age_medal_by_sport = pysql(query)
print(age_medal_by_sport)


# In[15]:


#Distinct number of sports palyed

query = """
SELECT DISTINCT Sport
FROM df_athlete
"""

distinct_sports = pysql(query)
print(distinct_sports)


# In[16]:


pysql("Select Distinct noc From df_athlete")


# In[17]:


#Top 15 Teams by Number of Years Participated in Olympics

query = """
SELECT Team, COUNT(DISTINCT Year) AS Years_Participated
FROM df_athlete
GROUP BY Team
ORDER BY Years_Participated DESC
Limit 15
"""

top_teams = pysql(query)

plt.figure(figsize=(10, 6))
plt.bar(top_teams['Team'], top_teams['Years_Participated'])
plt.title('Top 15 Teams by Number of Years Participated in Olympics')
plt.xlabel('Team')
plt.ylabel('Years Participated')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# In[18]:


pysql("Select count(Distinct Year) From df_athlete")


# In[19]:


#Unique players

query = """
SELECT Name, COUNT(DISTINCT ID) as ID_Count
FROM df_athlete
GROUP BY Name
HAVING ID_Count > 1
ORDER BY ID_Count DESC
LIMIT 10
"""

pysql(query)


# In[20]:


#total medals in each sport over 120 years top 15 teams

query = """
SELECT count(Distinct Medal) as Total_Medal, COUNT(DISTINCT Name) AS Unique_Players, Sport
From df_athlete
Where Medal is not null
Group by Sport
ORDER BY Total_Medal DESC
LIMIT 15
"""
pysql(query)


# In[21]:


#number of male and female in every year us 

query = """
SELECT COUNT(DISTINCT Name) AS Unique_Players, Sex, Year
From df_athlete
Where Team = "United States" 
Group by Year, Sex
ORDER BY Year

"""
us_gender_year=pysql(query)

us_gender_pivot = us_gender_year.pivot(index='Year', columns='Sex', values='Unique_Players').fillna(0)


# In[22]:


us_gender_pivot


# In[23]:


#Male and Female Participants from United States Over the Years

plt.figure(figsize=(10, 6))
plt.plot(us_gender_pivot.index, us_gender_pivot['M'], label='Male', marker='o')
plt.plot(us_gender_pivot.index, us_gender_pivot['F'], label='Female', marker='o')
plt.title('Male and Female Participants from United States Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Participants')
plt.legend(title='Gender')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


#number of participants from each team in 120 years

query = """
SELECT n.region AS Region, COUNT(DISTINCT a.Name) AS Participants
FROM df_athlete a
JOIN df_noc n
    ON a.NOC = n.NOC
GROUP BY n.region
ORDER BY Participants DESC
limit 15
"""

participants_by_region = pysql(query)


# In[25]:


participants_by_region


# In[26]:


#PArticipation years in Olympic of top 15 teams

# Drop rows with missing region
participants_by_region = participants_by_region.dropna(subset=['Region'])

# Sort for better readability
participants_by_region = participants_by_region.sort_values(by='Participants', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(participants_by_region['Region'], participants_by_region['Participants'])
plt.title('Olympic Participation Counts by Region')
plt.xlabel('Region')
plt.ylabel('Number of Participants')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# In[27]:


pysql("Select count(distinct NOC) from df_noc")


# In[28]:


df_athlete["Age"].describe()


# In[29]:


pysql("Select max(Age) from df_athlete")


# In[30]:


pysql("Select min(Age) from df_athlete")


# In[31]:


pysql("Select * From df_athlete where age = 97")


# In[32]:


pysql("Select * From df_athlete where age = 10")


# In[33]:


query = """
SELECT COUNT(DISTINCT Name) AS Unique_Players, Sex, Year
From df_athlete
Group by Year, Sex
ORDER BY Year

"""
us_gender_year=pysql(query)

us_gender_pivot = us_gender_year.pivot(index='Year', columns='Sex', values='Unique_Players').fillna(0)


# In[34]:


us_gender_pivot


# In[35]:


#correlation bet female ratio and years 

from scipy.stats import pearsonr
import numpy as np

# Make female ratio column
us_gender_pivot['Total'] = us_gender_pivot['F'] + us_gender_pivot['M']
us_gender_pivot['Female_Ratio'] = us_gender_pivot['F'] / us_gender_pivot['Total']

# Run correlation between Year and Female Ratio
years = us_gender_pivot.index.values.astype(float)
ratios = us_gender_pivot['Female_Ratio'].values.astype(float)

corr, pval = pearsonr(years, ratios)
print("Pearson correlation (Year vs Female Ratio):", round(corr, 3))
print("p-value:", pval)


# In[36]:


from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = us_gender_pivot.index.values.reshape(-1, 1)  # Years as X
y = us_gender_pivot['F'].values  # Female participants as y

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title("Female Participation in US Olympics Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Female Participants")
plt.legend()
plt.show()

# Coefficients
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


# In[37]:


future_year = np.array([[2028]])
pred_2028 = model.predict(future_year)
print("Predicted female participation in 2028:", int(pred_2028[0]))


# In[38]:


#correlation between age of aprticipants and preformance 
df_athlete['Performance'] = df_athlete['Medal'].map({'Gold': 1, 'Silver': 1, 'Bronze': 1}).fillna(0)
df_athlete['Performance']


# In[39]:


from scipy.stats import pearsonr

valid_data = df_athlete.dropna(subset=['Age', 'Performance'])
corr, p_value = pearsonr(valid_data['Age'], valid_data['Performance'])
print("Correlation:", corr, "P-value:", p_value)


# In[40]:


valid_data = df_athlete.dropna(subset=['Age', 'Performance'])
valid_data


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns

valid_data = df_athlete.dropna(subset=['Age', 'Performance'])

plt.figure(figsize=(8,6))
sns.scatterplot(x="Age", y="Performance", data=valid_data, alpha=0.3)
sns.regplot(x="Age", y="Performance", data=valid_data, scatter=False, color="red")

plt.title("Correlation between Age and Performance (Medal=1, No Medal=0)")
plt.xlabel("Age")
plt.ylabel("Performance (1=Medal, 0=No Medal)")
plt.show()


# In[42]:


#Textual Analysis for TF-IDF (Term Frequency-Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfVectorizer

# Build a small corpus: one "document" per Sport, concatenating its event names
docs = (df_athlete.dropna(subset=['Sport','Event'])
        .groupby('Sport')['Event'].apply(lambda s: ' '.join(map(str, s))).reset_index())

vec = TfidfVectorizer(lowercase=True, stop_words='english')
X = vec.fit_transform(docs['Event'])
terms = np.array(vec.get_feature_names_out())

# Top terms per sport
topk = 8
top_terms = {}
for i, sport in enumerate(docs['Sport']):
    row = X[i].toarray().ravel()
    idx = row.argsort()[::-1][:topk]
    top_terms[sport] = terms[idx].tolist()

# Example: print a few sports and their key terms
for s in list(top_terms.keys())[:5]:
    print(s, "→", top_terms[s])


# In[43]:


#chi aquare test - Is there an association between participants  gender and the type of sport they participate in?


from scipy.stats import chi2_contingency
query = """
Select Sport, Sex, count(Distinct Name) as pt
From df_athlete
group by Sport, Sex
"""
gender_sport = pysql(query)
contingency = gender_sport.pivot(index="Sport", columns="Sex", values="pt").fillna(0)

chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-Square Statistic:", chi2)
print("p-value:", p)


# In[44]:


# First: prepare the data
query = """
SELECT Sport, Sex, COUNT(DISTINCT Name) as Participants
FROM df_athlete
GROUP BY Sport, Sex
"""
sport_gender = pysql(query)

# Pivot for plotting
pivot_sport_gender = sport_gender.pivot(index="Sport", columns="Sex", values="Participants").fillna(0)

# --- Option 1: Stacked Bar Plot ---
pivot_sport_gender.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Male vs Female Participation by Sport")
plt.xlabel("Sport")
plt.ylabel("Number of Participants")
plt.xticks(rotation=90)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[45]:


query = """
SELECT Sport, Sex, COUNT(DISTINCT Name) as Participants
FROM df_athlete
GROUP BY Sport, Sex
"""
sport_gender = pysql(query)
pivot_sport_gender = sport_gender.pivot(index="Sport", columns="Sex", values="Participants").fillna(0)
pivot_sport_gender


# In[46]:


pysql("Select Sex, Sport From df_athlete Where Sport = 'Tug-Of-War'")


# In[47]:


query = """
SELECT Sport, Sex, COUNT(DISTINCT Name) AS Player_Count
FROM df_athlete
GROUP BY Sport, Sex
"""
gender_sport_counts = pysql(query)


# In[48]:


pivot = gender_sport_counts.pivot(index='Sport', columns='Sex', values='Player_Count').fillna(0)
pivot.columns = ['Female', 'Male']  # reorder if needed


# In[49]:


pivot['GSPI'] = pivot[['Male', 'Female']].min(axis=1) / pivot[['Male', 'Female']].max(axis=1)
pivot['GSPI']


# In[50]:


import matplotlib.pyplot as plt

pivot.sort_values('GSPI', ascending=False)['GSPI'].plot(kind='bar', figsize=(12,6))
plt.title("Gender-Sport Participation Index (GSPI) by Sport")
plt.ylabel("GSPI (0 = Skewed, 1 = Balanced)")
plt.xlabel("Sport")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[51]:


#Q3

from scipy.stats import f_oneway
import pandasql as psql

# Step 1: Get heights grouped by sport using pandasql
query = """
SELECT Sport, Height
FROM df_athlete
WHERE Height IS NOT NULL
"""
heights_by_sport = psql.sqldf(query, locals())

# Step 2: Prepare data for ANOVA
groups = [group["Height"].values for name, group in heights_by_sport.groupby("Sport")]

# Step 3: Run ANOVA
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)


# In[ ]:





# In[52]:


#q4


# In[53]:


query = """
SELECT Team, Season, COUNT(Medal) AS Total_Medals
FROM df_athlete
WHERE Medal IS NOT NULL
GROUP BY Team, Season
"""
medals_by_season = pysql(query)


# In[54]:


pivot_medals = medals_by_season.pivot(index='Team', columns='Season', values='Total_Medals').fillna(0)
pivot_medals.head()


# In[55]:


from scipy.stats import pearsonr

corr, p_value = pearsonr(pivot_medals['Summer'], pivot_medals['Winter'])
print("Correlation:", corr)
print("p-value:", p_value)


# In[56]:


#q5


# In[57]:


import pandas as pd

# Define bins and labels
bins = [0, 18, 25, 30, 35, 40, 100]
labels = ['<18', '18-25', '26-30', '31-35', '36-40', '>40']

df_athlete['Age_Group'] = pd.cut(df_athlete['Age'], bins=bins, labels=labels, right=False)


# In[58]:


df_athlete['Medal_Status'] = df_athlete['Medal'].notnull().astype(int)


# In[59]:


from pandasql import sqldf
pysql = lambda q: sqldf(q, globals())

query = """
SELECT Age_Group, Medal_Status, COUNT(*) as Count
FROM df_athlete
WHERE Age_Group IS NOT NULL
GROUP BY Age_Group, Medal_Status
"""

contingency_data = pysql(query)
print(contingency_data)


# In[60]:


contingency_table = contingency_data.pivot(index='Age_Group', columns='Medal_Status', values='Count').fillna(0)
print(contingency_table)


# In[61]:


from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)


# In[62]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr, f_oneway

# Create Performance column (1 = Medal, 0 = No Medal)
df_athlete['Performance'] = np.where(df_athlete['Medal'].notnull(), 1, 0)

# Drop missing ages
valid_data = df_athlete.dropna(subset=['Age', 'Performance'])

# --- Pearson Correlation ---
corr, p_value = pearsonr(valid_data['Age'], valid_data['Performance'])
print("Pearson Correlation:", corr)
print("P-value:", p_value)


# In[ ]:




