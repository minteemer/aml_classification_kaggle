import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing

TEST_SIZE = 0.3

# Make Y and X train and test sets
train = pd.read_csv('train.csv')
Y = train['OpenStatus']
# X_test = pd.read_csv('test.csv')
X = train.drop('OpenStatus', axis=1)

# TODO: analyze dates
# But for now simply drop them
X = X.drop('PostClosedDate', axis=1) \
    .drop('PostCreationDate', axis=1) \
    .drop('OwnerUserId', axis=1) \
    .drop('PostId', axis=1)


# prepare account creation dates
def date_string_to_timestamp(date):
    try:
        return datetime.strptime(date, "%m/%d/%Y %H:%M:%S").timestamp()
    except ValueError:  # some dates are in different format
        return datetime.strptime(date, "%Y-%m-%d").timestamp()


creation_timestamps = X["OwnerCreationDate"].map(date_string_to_timestamp)
X["OwnerCreationDate"] = (creation_timestamps - creation_timestamps.mean()) / creation_timestamps.std()

# TODO: analyze titles and bodyMarkdowns considering tags and not only
# But for now simply drop them
X = X.drop('Title', axis=1).drop('BodyMarkdown', axis=1)

# Creating new column counting tags (more tags -> better post)
X['Tag1'] = X['Tag1'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag2'] = X['Tag2'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag3'] = X['Tag3'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag4'] = X['Tag4'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag5'] = X['Tag5'].map(lambda x: 0 if isinstance(x, float) else 1)
count_tags = []
for index, row in X.iterrows():
    count_tags.append(row['Tag1'] + row['Tag2'] + row['Tag3'] + row['Tag4'] + row['Tag5'])
X['CountTags'] = count_tags
X = X.drop('Tag1', axis=1).drop('Tag2', axis=1).drop('Tag3', axis=1) \
    .drop('Tag4', axis=1).drop('Tag5', axis=1)


def get_normalised_data():
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
    return train_test_split(pd.DataFrame(X_scaled), Y, test_size=TEST_SIZE)


def get_data():
    return train_test_split(X, Y, test_size=TEST_SIZE)
