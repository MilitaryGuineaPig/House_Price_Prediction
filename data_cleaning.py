import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(data):
    data.dropna(inplace=True)

    neighbourhood_group_num = LabelEncoder()
    neighbourhood_num = LabelEncoder()
    room_type_num = LabelEncoder()

    neighbourhood_group_num.fit(data['neighbourhood_group'])
    neighbourhood_num.fit(data['neighbourhood'])
    room_type_num.fit(data['room_type'])

    data['neighbourhood_group'] = neighbourhood_group_num.transform(data['neighbourhood_group'])
    data['neighbourhood'] = neighbourhood_num.transform(data['neighbourhood'])
    data['room_type'] = room_type_num.transform(data['room_type'])

    drop_column = ['name', 'host_name', 'last_review']
    data = data.drop(drop_column, axis=1)

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data_scaled
