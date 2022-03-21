### Đề xuất đơn giản, gợi ý 20 bộ phim được đánh giá cao nhất ###

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv/movies_metadata.csv', low_memory=False)     #read_csv(): lấy dữ liệu đầu vào

# Print the first three rows
# print(metadata.head(3))

# Calculate mean of vote average column
C = metadata['vote_average'].mean()     #mean(): Trả về giá trị trung bình của các giá trị trên trục được yêu cầu.
# print(C)    

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)     #quantile(): Trả về các giá trị tại lượng tử đã cho qua trục được yêu cầu.
# print(m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]     #copy(): Tạo một bản sao với chỉ số và dữ liệu tương đương của đối tượng / loc[]: Truy cập một nhóm hàng và cột hoặc một mảng boolean.
# print(q_movies.shape)

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)     #apply(): thêm một hàm dọc theo trục của DataFrame. /axis=0là chiều hướng xuống dưới và chiều hướng axis=1 hướng sang phải.

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)   #sort_values(): Sắp xếp theo các giá trị dọc tăng hoặc giảm

#Print the top 15 movies
top20 = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)     #head(): lấy ra 20 hàng đầu tiên
print(top20)

