from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = [GaussianNB(),DecisionTreeClassifier(max_depth=5)]
acc=[0,0]

classes = {
    0: "Iris Setosa",
    1: "Iris Versicolour",
    2: "Iris Virginica"
}

def load_model():
	X, y = datasets.load_iris(return_X_y=True)

	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
	clf[0].fit(X_train, y_train)
	clf[1].fit(X_train, y_train)

	acc[0] = accuracy_score(y_test, clf[0].predict(X_test))
	acc[1] = accuracy_score(y_test, clf[1].predict(X_test))
	print(f"Model trained with accuracy: {round(acc[acc.index(max(acc))], 3)} \n in comparision to accuracy: {round(acc[acc.index(min(acc))], 3)}")

def predict(query_data):
	x = list(query_data.dict().values())
	prediction = clf[acc.index(max(acc))].predict([x])[0] 
	print(f"Model prediction: {classes[prediction]}")
	return classes[prediction]




