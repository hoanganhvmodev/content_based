### Đề xuất dựa trên nội dung ###

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv/movies_metadata.csv', low_memory=False)      #read_csv(): lấy dữ liệu đầu vào

#Print plot overviews of the first 5 movies.
print(metadata[['title', 'overview']].head(5))      #head(): lấy tiêu đề và giởi thiệu tổng quan của 5 bộ phim đầu tiên

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer     #TfidfVectorizer TF-IDF: hàm tính tài liệu nghịch đảo

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')      #fillna(): phương thức thay thế các giá trị NULL bằng một giá trị được chỉ định.

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])    #fit_transform(): Chuyển đổi tài liệu thô sang một ma trận các tính năng TF-IDF.

#Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)

#Array mapping from feature integer indices to feature name.
fName = tfidf.get_feature_names_out()[5000:5010]    #get_feture_names_out(): Trả về danh sách các tên đối tượng, được sắp xếp theo chỉ số của chúng.
print(fName)

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)      #linear_kernel(): Tính hạt nhân tuyến tính giữa X và Y.
# print(cosine_sim.shape)
# print(cosine_sim[1])

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()  #pd.Series(): Tạo một Chuỗi pandas đơn giản từ danh sách, drop_duplicates(): Trả lại DataFrame với các hàng trùng lặp đã bị loại bỏ.
# print(indices[:10])

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))   #list(): Tạo danh sách / enumerate(): thêm một bộ đếm vào một đối tượng có thể lặp lại và trả về nó ở dạng liệt kê đối tượng.

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)   #sorted(): trả về một danh sách được sắp xếp của đối tượng có thể lặp được chỉ định.

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:10]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    print(metadata['title'].iloc[movie_indices])
    return metadata['title'].iloc[movie_indices]

get_recommendations('Batman Forever')
