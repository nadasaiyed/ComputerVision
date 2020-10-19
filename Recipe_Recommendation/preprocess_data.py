#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


print("Reading Ingredients dataset")
#loading the ingredients from https://dominikschmidt.xyz/simplified-recipes-1M/
with np.load('simplified-recipes-1M.npz',allow_pickle=True) as recipe:
    ingredients = recipe['ingredients']

#Preprocessing, simplifying the ingredients' list further

remove = [x for x in ingredients if x.endswith('ed') and not 'seed' in x and not 'weed' in x and not ' red' in x or 'temperature' in x or 'any'==x or 'natural' in x]
remove.append([item for item in ingredients if len(item.split())>1 and len(item.split())<3 and item.split()[0].endswith('ed') and not (item.split()[0] == 'red' or item.split()[0] == 'whipped') and item.split()[1] in ingredients])
# print(remove)
print("\t Number of Ingredients : ",len(ingredients))
ingredients = [x for x in ingredients if x not in remove]
remove = [x for x in ingredients if len(x)<3]
ingredients = [x for x in ingredients if x not in remove]
ingredients.append('tabasco')
ingredients.append('jalapeo') # because our dataset contains incorrect spelling of jalapeno
print("\tNumber of Ingredients after processing: ",len(ingredients))


#Recipe dataset
print("Reading Recipe dataset")
data = pd.read_json('full_format_recipes.json')
print("\tAttributes of the dataset: ",data.columns)
data.head()


data = data.drop(columns=['date', 'rating'])
data = data.dropna(subset=['title','ingredients','directions'])
#dropping N/A values for columns
print("\tNumber of records in the dataset",len(data))


# In[58]:


title=data[data['title'].duplicated()]['title']
data[data['title'].isin(title)].sort_values('title') # 2336*2 rows duplicates

#drop duplicated values

data.drop_duplicates(subset ="title",keep = 'first', inplace = True) 
print("\tDuplicate titles: ",len(title))
len(data)

# Exploring recipe's nutritional values

nutrients = data[['fat','calories','sodium','protein']].dropna()
sns.set(style="whitegrid")
sns.boxplot(x = 'variable', y = 'value' , data = pd.melt(nutrients)).set_title('Boxplot for food nutrient values')

# ax = sns.boxplot(x=data['calories'].notnull())
print()
print("Description of nutrients")
print(nutrients.describe())

# Extracting ingredients from noise ingredient list

def get_ingredients(document):
    items=[]
    unparsed=[]
    for sentence in document:
        sentence=re.sub(r'[^a-zA-Z -]+', '', sentence).strip().lower()
        item_list=[]
        try:
            for item in ingredients:
                if (item in sentence):
                    item_list.append(item)
            item_list.sort(key=len,reverse=True)
            items.append(item_list[0])
            if(len(item_list)>1 and item_list[1] not in item_list[0] and item_list[1] not in 'ice'):  
                items.append(item_list[1])
        except:
            pass
    return items


# Gettign distinct ingredients

def get_unique_items(l):
    s=set(l)
    return [x for x in s]


print()
print("Processing the dataset...")

# data['ingredients_list']
data['ingredients_list'] = data['ingredients'].apply(get_ingredients)
data['ingredients_list'] = data['ingredients_list'] .apply(get_unique_items)
data['ingredients_count']=data['ingredients_list'].apply(len) # Count of each ingredient
print("\t Removing items which have no ingredients. Removed items:",len(data[data['ingredients_count']==0]))
data = data[data['ingredients_count']>0]
data['ingredients_doc']=data.ingredients_list.str.join(' ')

# Stemming the ingredients to avoid plural forms

from nltk.stem import PorterStemmer
porter = PorterStemmer()

data['ingredients_stem']=data['ingredients_doc']
for i in range(len(data)):
    x = (data['ingredients_stem'].iloc[i].split(' '))
    data['ingredients_stem'].iloc[i] = ' '.join([porter.stem(a) for a in x])

# Removing unnecessary words from stemmed list

data['ingredients_stem']=data['ingredients_stem'].str.replace('freshli','').str.replace('fresh','').str.replace('ground','').str.replace('dri ','').str.replace('chop','').str.replace('grate ','').str.replace('tea ','')
#tea for teaspoon


print(data.shape)
print("Saving the processed dataset")
data.to_csv("processed_data.csv",index=False)