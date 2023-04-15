import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/Users/monkey/Public/Python/Jop_prepar/Task_1/archive/AB_NYC_2019.csv")
pd.set_option('display.max_columns', None)
columns_name = data.columns
# print(columns_name) # to see the data

sns.pairplot(data)
# display the plot
plt.show()

# print(data.tail()) view data
data.dropna(inplace=True)

# split data for training
Xdata = data.drop(['name', 'price', 'host_name', 'last_review'], axis=1)
# print(Xdata) # check
ydata = data['price']
# print(ydata) # check

# convert data to nums
# create a LabelEncoder object
neighbourhood_group_num = LabelEncoder()
neighbourhood_num = LabelEncoder()
room_type_num = LabelEncoder()
# fit the LabelEncoder object to the state names
neighbourhood_group_num.fit(Xdata['neighbourhood_group'])
neighbourhood_num.fit(Xdata['neighbourhood'])
room_type_num.fit(Xdata['room_type'])
# transform the state names into numerical labels
Xdata['neighbourhood_group'] = neighbourhood_group_num.transform(Xdata['neighbourhood_group'])
Xdata['neighbourhood'] = neighbourhood_num.transform(Xdata['neighbourhood'])
Xdata['room_type'] = room_type_num.transform(Xdata['room_type'])

print(Xdata)



