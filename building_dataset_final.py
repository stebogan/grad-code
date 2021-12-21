#%% Modules, Classes and Functions
import numpy as np
from sklearn.model_selection import KFold
import torch
from matplotlib import pyplot as plt
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
# Classification of Building data

# Logistic Regression driver code
class LogisticRegression(torch.nn.Module):
    def __init__(self, D_in, H, D_out, num_hidden_layers):
        super(LogisticRegression, self).__init__()
        
        self.input_layer = torch.nn.Linear(D_in, H)
        
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(H, H))
            
        self.output_layer = torch.nn.Linear(H, D_out)

    def forward(self, x):
        
        y_pred  = torch.nn.functional.sigmoid(self.input_layer(x))
        
        for layer in self.hidden_layers:
            y_pred = torch.nn.functional.sigmoid(layer(y_pred))
            
        y_pred   = self.output_layer(y_pred)
        return y_pred
    
    
def train(x_train, y_train, H, learning_rate, epoch, nhl, wd):
    model = LogisticRegression(x_train.shape[1], H, 3, nhl)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = wd)
    a = np.zeros(len(range(epoch)))
    b = np.zeros(len(range(epoch)))
    for epoch in range(epoch):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(y_train).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.squeeze().long())

        loss.backward() # calculate gradients
        
        optimizer.step() # updates weights
        print('epoch {}: loss = {}'.format(epoch, loss.item()))
        a[epoch] = epoch
        b[epoch] = loss.item()
    plt.figure()
    plt.plot(a,b)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    return model


# K-fold cross validation
def kfold_CV(X, Y, K, H, learning_rate, epochs, NHL, wd):
    kf = KFold(n_splits = K, shuffle = True)
    test_accuracy = np.empty(0)
    Y = Y.reshape(-1,1)
    for trn_idx, tst_idx in kf.split(X):
        x_train, x_test = X[trn_idx, :], X[tst_idx, :]
        y_train, y_test = Y[trn_idx, :], Y[tst_idx, :]
        
        md1 = train(x_train, y_train, H, learning_rate, epochs, NHL, wd = wd)
        with torch.no_grad():
            y_predicted = md1(torch.from_numpy(x_test).float())
        
        # Select class with highest predicition value
        _, pred = torch.max(y_predicted, 1)
        test_accuracy = np.append(test_accuracy, classification_results(pred.numpy(), y_test[:, 0]))
        
    return test_accuracy.mean()
        

def error_metrics(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mape = np.mean(np.abs(y_pred - y_test) / y_test)
    mae = np.mean(np.abs(y_pred - y_test))
    mbe = np.mean(y_pred - y_test)
    # r2 = np.corrcoef(y_pred.squeeze(), y_test.squeeze())[0, 1]**2
    return rmse, mape, mae, mbe


# Calculate predicitve accuracy
def classification_results(y_predicted,y_actual):
    print(np.vstack((y_predicted,y_actual)))
    return sum(y_predicted == y_actual) / len(y_predicted)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    
#%% Import CSV Data and Process

filename = 'Building_Load_Dataset.csv'

fields = []
rows = [] 

with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    
    for row in csvreader:
        rows.append(row)
        
# Create storage for features and target
data = np.zeros((8216,7))
date = np.zeros((8216,4))
target = np.zeros((8216,1))

# Get date data
for i in range(len(rows)):
    # Hour of Day
    date[i][0] = int(rows[i][0].split('/')[2].split(' ')[1].split(':')[0])*60 + int(rows[i][0].split('/')[2].split(' ')[1].split(':')[1])
    
    # Day (counting up, no repeats)
    if int(rows[i][0].split()[0].split('/')[0]) == 6:
        date[i][1] = int(rows[i][0].split()[0].split('/')[1])
    elif int(rows[i][0].split()[0].split('/')[0]) == 7:
        date[i][1] = int(rows[i][0].split()[0].split('/')[1]) + 30
    elif int(rows[i][0].split()[0].split('/')[0]) == 8:
        date[i][1] = int(rows[i][0].split()[0].split('/')[1]) + 60
    else:
        continue
    # Month
    date[i][2] = int(rows[i][0].split()[0].split('/')[0])-6
    # Weekend ID (0 if weekday, 1 if weekend)
    if np.any(date[i][1] == [6,7,13,14,20,21,27,28,34,35,41,42,48,49,55,56,62,63,69,70,76,77,83,84,90,91]):
        date[i][3] = 1
    else:
        date[i][3] = 0

# Rest of data
for i in range(len(rows)):
    for j in range(1,8):
        if rows[i][j] == '':
            rows[i][j] = rows[i-1][j]
            data[i][j-1] = rows[i][j]
        else:
            data[i][j-1] = rows[i][j]
         
fields = ['Hour of Day', 'Barometer', 'Hydrometer', 'RainGauge', 'Solar Radiation Sensor', 'Thermometer', 'Wind Vane', 'Energy Load']
data = np.concatenate((date, data), axis=1)

# Make target month
target[:,0] = data[:,2]
data = np.delete(data,[1,2,3],1)

# Target Feature Correlations
corrcoef_list = np.empty(shape=(1,data.shape[1]))
for j in range(data.shape[1]):
    corrcoef_list[:,j] = np.corrcoef(x=data[:, j],y = target[:,0])[0,1] # Get list of data.shape[1] corrcoef values
    
# Feature Correlations
feature_corrcoef_list = np.empty(shape=(data.shape[1],data.shape[1]))
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        feature_corrcoef_list[i,j] = np.corrcoef(x=data[:, i],y=data[:,j])[0,1]

# Scale features
scaler = MinMaxScaler()
data = scaler.fit_transform(data) 
#%% A few plots

plt.figure()
plt.plot(data[:, 4][target[:, 0] == 0], data[:, 5][target[:, 0] == 0], 'o', markersize=5)
plt.plot(data[:, 4][target[:, 0] == 1], data[:, 5][target[:, 0] == 1], 'o', markersize=5)
plt.plot(data[:, 4][target[:, 0] == 2], data[:, 5][target[:, 0] == 2], 'o', markersize=5)
plt.xlabel('solar radiation [W/m2]')
plt.ylabel('temperature [F]')
plt.title('solar radiation vs. temperature')
plt.legend(['June', 'July', 'August'])
plt.show()

plt.figure()
plt.plot(data[:, 2][target[:, 0] == 0], data[:, 3][target[:, 0] == 0], 'o', markersize=5)
plt.plot(data[:, 2][target[:, 0] == 1], data[:, 3][target[:, 0] == 1], 'o', markersize=5)
plt.plot(data[:, 2][target[:, 0] == 2], data[:, 3][target[:, 0] == 2], 'o', markersize=5)
plt.xlabel('solar radiation [W/m2]')
plt.ylabel('temperature [F]')
plt.title('solar radiation vs. temperature')
plt.legend(['June', 'July', 'August'])
plt.show()

for q in range(data.shape[1]-1):
    plt.figure()
    plt.plot(data[:, q][target[:, 0] == 0], data[:, 7][target[:, 0] == 0], 'o', markersize=5)
    plt.plot(data[:, q][target[:, 0] == 1], data[:, 7][target[:, 0] == 1], 'o', markersize=5)
    plt.plot(data[:, q][target[:, 0] == 2], data[:, 7][target[:, 0] == 2], 'o', markersize=5)
    plt.xlabel('%s' % fields[q])
    plt.ylabel('energy use [kW]')
    plt.title('%s vs. energy use' % fields[q])
    plt.legend(['June', 'July', 'August'])
    plt.show()

# Histograms
for c in range(8):
    plt.figure()
    plt.hist(data[:,c][target[:, 0] == 0], bins = 50)
    plt.hist(data[:,c][target[:, 0] == 1]+1, bins = 50)
    plt.hist(data[:,c][target[:, 0] == 2]+2, bins = 50)
    plt.title('Histogram of Normalized %s Feature (Offset by +1 each month)' % fields[c])
    plt.ylabel('# occurnences')
    plt.xlabel('%s value' % fields[c])
    plt.legend(['June', 'July', 'August'])
    plt.show()

#%% LG: Grid Search all Variables==============================================

# Tune number of neurons, learning rate, and weight decay
# Cross-validation: 3-fold

H_list = [200,240,280,320]
lr_list = [0.01] #[10**i for i in range(-3, -1)]
wd_list = [0.00001]#[10**i for i in range(-7, -5)]

# Create 3-D array for accuracy info
accuracy = np.zeros((len(H_list), len(lr_list), len(wd_list)))

# Iterate through grid values
for h, H in enumerate(H_list):
    for l, lr in enumerate(lr_list):
        for w, wd in enumerate(wd_list):
            test_accuracy = kfold_CV(X = data, Y = target[:,0], K = 2, H = H, learning_rate = lr, epochs = 4000, NHL = 4, wd = wd)
            
            # Update accuracy array
            accuracy[h, l, w] = test_accuracy
            
            print('H = {}, lr = {}, wd =  {}: Accuracy = {}'.format(H, lr, wd, test_accuracy))
        
i, j, k = np.argwhere(accuracy == np.max(accuracy))[0] # (i,j,k) location of best accuracy

h_best = H_list[i]
lr_best = lr_list[j]
wd_best = wd_list[k]  

# Train model with best parameters=============================================
np.random.seed(143)
idx = np.random.permutation(data.shape[0])
trn_idx, tst_idx = idx[:6900], idx[6900:]

# Define training and testing data sets
x_train, x_test = data[trn_idx, :], data[tst_idx, :]
y_train, y_test = target[trn_idx, :], target[tst_idx, :]

# model = train(x_train, y_train, h_best, lr_best, 4000, 4, wd_best)
model = train(x_train, y_train, 200, lr_best, 4000, 2, wd_best)

with torch.no_grad():
    y_predicted = model(torch.from_numpy(x_test).float())

_, pred = torch.max(y_predicted, 1)
pred = pred.numpy().reshape(-1,1)

print(classification_results(pred, y_test))
cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['June','July','August'])
disp.plot()
plt.title('Confusion Matrix: 4 hidden layer, H = %r, lr = %r' % (h_best, lr_best))
plt.set_cmap('jet')
plt.show()

#%% LG: Grid Search 2 all Variables============================================

# Tune number of neurons, learning rate, and weight decay
# Cross-validation: 3-fold

H_list = [100, 150, 200]
nhl_list = [2,3,4]
epoch_list = [3000,3500,4000]

# Create 3-D array for accuracy info
accuracy = np.zeros((len(H_list), len(lr_list), len(wd_list)))

# Iterate through grid values
for h, H in enumerate(H_list):
    for n, nhl in enumerate(nhl_list):
        for e, epoch in enumerate(epoch_list):
            test_accuracy = kfold_CV(X = data, Y = target[:,0], K = 3, H = H, learning_rate = 0.01, epochs = epoch, NHL = nhl, wd = 1e-6)
            
            # Update accuracy array
            accuracy[h, l, w] = test_accuracy
            
            print('H = {}, lr = {}, wd =  {}: Accuracy = {}'.format(H, lr, wd, test_accuracy))
        
i, j, k = np.argwhere(accuracy == np.max(accuracy))[0] # (i,j,k) location of best accuracy

h_best = H_list[i]
nhl_best = nhl_list[j]
epoch_best = epoch_list[k]  

# Train model with best parameters=============================================
np.random.seed(143)
idx = np.random.permutation(data.shape[0])
trn_idx, tst_idx = idx[:6900], idx[6900:]

# Define training and testing data sets
x_train, x_test = data[trn_idx, :], data[tst_idx, :]
y_train, y_test = target[trn_idx, :], target[tst_idx, :]

# model = train(x_train, y_train, h_best, 0.01, epoch_best, nhl_best, 1e-6)
model = train(x_train, y_train, 100, 0.01, 3400, 3, 1e-6)
with torch.no_grad():
    y_predicted = model(torch.from_numpy(x_test).float())

_, pred = torch.max(y_predicted, 1)
pred = pred.numpy().reshape(-1,1)

print(classification_results(pred, y_test))
cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['June','July','August'])
disp.plot()
plt.title('Confusion Matrix: %r hidden layer, H = %r, lr = %r' % (nhl_best, h_best, lr_best))
plt.set_cmap('jet')
plt.show()

#%% LG: Grid search (weather variables)========================================
weather = np.delete(data,[0,7],1)

H_list = [20,50,100,200]
lr_list = [10**i for i in range(-4, -1)]
wd_list = [10**i for i in range(-7, -4)]

accuracy = np.zeros((len(H_list), len(lr_list), len(wd_list)))


for h, H in enumerate(H_list):
    for l, lr in enumerate(lr_list):
        for w, wd in enumerate(wd_list):
            test_accuracy = kfold_CV(X = weather, Y = target[:,0], K = 3, H = H, learning_rate = lr, epochs = 3000, NHL = 2, wd = wd)
            
            accuracy[h, l, w] = test_accuracy
            
            print('H = {}, lr = {}, wd =  {}: Accuracy = {}'.format(H, lr, wd, test_accuracy))
        
i, j, k = np.argwhere(accuracy == np.max(accuracy))[0]

h_best = H_list[i]
lr_best = lr_list[j]
wd_best = wd_list[k]  

# Train model with best parameters=============================================
np.random.seed(1234)
idx = np.random.permutation(weather.shape[0])
trn_idx, tst_idx = idx[:6900], idx[6900:]

# Define training and testing data sets
x_train, x_test = weather[trn_idx, :], weather[tst_idx, :]
y_train, y_test = target[trn_idx, :], target[tst_idx, :]

model = train(x_train, y_train, h_best, lr_best, 3000, 2, wd_best)

with torch.no_grad():
    y_predicted = model(torch.from_numpy(x_test).float())

_, pred = torch.max(y_predicted, 1)
pred = pred.numpy().reshape(-1,1)

print(classification_results(pred, y_test))
cm = confusion_matrix(y_test, pred)

plt.figure()
plt.plot(y_test, 'o', label='test', markersize = 10)
plt.plot(pred,'o', label='predicted')
plt.title('Test vs. Prediction Class Identification Comparison')
plt.ylabel('month')
plt.xlabel('arbitrary sample point')
plt.legend()


#%% SVC 

# Split into test/train
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Train SVC
model = svm.SVC(C=300,kernel='rbf',gamma=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Error metrics
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=['June','July','August'],
    cmap=plt.cm.Blues,
    normalize=None
)
disp.ax_.set_title('Confusion matrix')
print(disp.confusion_matrix)
plt.show()

# Hyperparameter tuning
param_grid = {
    'C': [10,20,50,70,90,100,120,150,160,170,200],
    'gamma': [0.01,0.1,1,10,20,30,40,50,60],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, verbose=3)
grid.fit(X_train, np.ravel(y_train))

# Best hyperparameters
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(X_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['June','July','August'])
disp.plot()
plt.title('Confusion Matrix: C = %r, gamma = %r, kernel = %r' % (grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['kernel']))
plt.set_cmap('jet')
plt.show()

#%% PCA SVC
X = data
y = np.ravel(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pca = PCA(n_components=5)# adjust yourself
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'C': [10,20,50,70,90,100,120,150,160,170,200],
    'gamma': [0.01,0.1,1,10,20,30,40,50,60],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, verbose=3)
grid.fit(X_t_train, y_train)

# Best hyperparameters
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(X_t_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['June','July','August'])
disp.plot()
plt.title('Confusion Matrix: C = %r, gamma = %r, kernel = %r' % (grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['kernel']))
plt.set_cmap('jet')
plt.show()

plt.figure()
q = 0
w = 1
plt.plot(X_t_train[:, q][y_train == 0], X_t_train[:, w][y_train == 0], 'o', markersize=5)
plt.plot(X_t_train[:, q][y_train == 1], X_t_train[:, w][y_train == 1], 'o', markersize=5)
plt.plot(X_t_train[:, q][y_train == 2], X_t_train[:, w][y_train == 2], 'o', markersize=5)
plt.legend(['June', 'July', 'August'])
plt.title("PCA x%r vs. x%r" % ((w+1),(q+1)))
plt.xlabel("x%r" % (q+1))
plt.ylabel("x%r" % (w+1))

#%% NCA SVC
X = data
y = np.ravel(target)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
nca = NeighborhoodComponentsAnalysis(n_components=5)# adjust yourself
nca.fit(X,y)
X_t = nca.transform(X)
scaler = MinMaxScaler()
X_t = scaler.fit_transform(X_t)
X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=0)



# Hyperparameter tuning
param_grid = {
    'C': [100,120,150,160,180,200,300],
    'gamma': [0.0001,0.001,0.01,0.1,1,10,100,200,300,500],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, verbose=3)
grid.fit(X_t_train, y_train)

# Best hyperparameters
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(X_t_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['June','July','August'])
disp.plot()
plt.title('Confusion Matrix: C = %r, gamma = %r, kernel = %r' % (grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['kernel']))
plt.set_cmap('jet')
plt.show()

plt.figure()
q = 1
w = 4
plt.plot(X_t_train[:, q][y_train == 0], X_t_train[:, w][y_train == 0], 'o', markersize=5)
plt.plot(X_t_train[:, q][y_train == 1], X_t_train[:, w][y_train == 1], 'o', markersize=5)
plt.plot(X_t_train[:, q][y_train == 2], X_t_train[:, w][y_train == 2], 'o', markersize=5)
plt.legend(['June', 'July', 'August'])
plt.title("NCA x%r vs. x%r" % ((w+1),(q+1)))
plt.xlabel("x%r" % (q+1))
plt.ylabel("x%r" % (w+1))
