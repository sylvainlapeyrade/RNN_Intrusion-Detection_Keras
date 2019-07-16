import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split

# Must declare data_dir as the directory of training and test files
# data_dir="./datasets/KDD-CUP-99/"
data_dir = "./data/"
raw_data_filename = data_dir + "kddcup.traindata.corrected.csv"
# raw_data_filename = data_dir + "kddcup.data_10_percent"

print("Loading raw data")

raw_data = pd.read_csv(raw_data_filename, header=None)

print("Transforming data")
# Categorize columns: "protocol", "service", "flag", c"attack_type"
raw_data[1], protocols = pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])
raw_data[3], flags = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features = raw_data.iloc[:, :raw_data.shape[1]-1]
labels = raw_data.iloc[:, raw_data.shape[1]-1:]

# convert them into numpy arrays
# features= numpy.array(features)
# labels= numpy.array(labels).ravel() # this becomes an 'horizontal' array
labels = labels.values.ravel()  # this becomes a 'horizontal' array

# TODO: get features names and target name

# Separate data in train set and test set
df = pd.DataFrame(features)
# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling
# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(
    df, labels, train_size=0.8, test_size=0.2)
print("X_train, y_train:", X_train.shape, y_train.shape)
print("X_test, y_test:", X_test.shape, y_test.shape)


# Training, choose model by commenting/uncommenting clf=
print("Training model")
# , max_features=0.8, min_samples_leaf=3, n_estimators=500,
# min_samples_split=3, random_state=10, verbose=1)
clf = RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102)
# clf = DecisionTreeClassifier(criterion='gini', splitter='best',
#  max_depth=None, min_samples_split=2, min_samples_leaf=1,
#  min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
#  max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None,
#  presort=False)

trained_model = clf.fit(X_train, y_train)

print("Score: ", trained_model.score(X_train, y_train))

# Predicting
print("Predicting")
y_pred = clf.predict(X_test)

print("Computing performance metrics")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print("Confusion matrix:\n", results)
print("Error: ", error)
