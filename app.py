from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form['classifier']
    param1 = request.form['param1']
    param2 = request.form['param2']
    param3 = request.form['param3']

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = get_classifier_instance(classifier_name, param1, param2, param3)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    img_str = get_confusion_matrix_base64_img(confusion_matrix, y)

    return render_template('resultado.html', accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score, confusion_matrix=img_str)

def get_classifier_instance(classifier_name, param1, param2, param3):
    match classifier_name:
        case 'KNN':
            return KNeighborsClassifier(n_neighbors=int(param1), leaf_size=int(param2), n_jobs=int(param3))
        case 'MLP':
            return MLPClassifier(max_iter=int(param1), alpha=int(param2), max_fun=int(param3))
        case 'DT':
            return DecisionTreeClassifier(max_depth=int(param1), random_state=int(param2), max_leaf_nodes=int(param3))
        case 'RF':
            return RandomForestClassifier(n_estimators=int(param1), max_depth=int(param2), random_state=int(param3))

def get_confusion_matrix_base64_img(confusion_matrix, y):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(set(y))), set(y))
    plt.yticks(range(len(set(y))), set(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_b64_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()

    return img_b64_str


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

if __name__ == '__main__':
    app.run(debug=True)
