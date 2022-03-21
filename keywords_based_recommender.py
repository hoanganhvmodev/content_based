#### Đề xuất Dựa trên Từ khóa ###

# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np

#Import function get_recommendations
import content_based_recommentdataions as cbr

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv/movies_metadata.csv', low_memory=False)      #read_csv(): lấy dữ liệu đầu vào

# Load keywords and credits
credits = pd.read_csv('credits.csv/credits.csv')         #read_csv(): lấy dữ liệu đầu vào
keywords = pd.read_csv('keywords.csv/keywords.csv')      #read_csv(): lấy dữ liệu đầu vào


# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')       #astype(): truyền một đối tượng vào dtype được chỉ định
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')     #merge(): gộp DataFrame hoặc các đối tượng được đặt tên với một phép nối csdl
metadata = metadata.merge(keywords, on='id')

# Print the first two movies of your newly merged metadata
# print(metadata.head(2))

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)   #apply(lintear_eval): lấy giá trị chuỗi input()và chuyển đổi nó thành int

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Print the new features of the first 3 films
# print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):      #isinstance(): kiểm tra kiểu dl có đúng với kiểu của tham số
            return str.lower(x.replace(" ", ""))        #lowre(): chuyển string sang kiểu chữ thường/ replace(): Thay thế tất cả sự xuất hiện của 1 từ hoặc khoảng trống:
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)        #apply(): thêm một hàng dọc theo trục của DataFrame

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])      #join(): nhận các mục và nối thành một chuỗi

# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)      
# print(metadata[['soup']].head(2))

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')       #countVectorizer(): Chuyển đổi một tập hợp các tài liệu văn bản thành một ma trận số lượng mã thông báo.
count_matrix = count.fit_transform(metadata['soup'])        #fit_transform(): Chuyển đổi tài liệu thô sang một ma trận các tính năng TF-IDF.
# print(count_matrix.shape)

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity  
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)     #cosine_similarity(): Tính độ tương thích cosin giữa các mẫu trong X và Y.(độ giống nhau)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()       #reset_index(): Đặt lại chỉ mục hoặc một cấp của đối tượng
indices = pd.Series(metadata.index, index=metadata['title'])        #pd.Series(): Tạo một Chuỗi pandas đơn giản từ danh sách

cbr.get_recommendations('The Godfather', cosine_sim2)
