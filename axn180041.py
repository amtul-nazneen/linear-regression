#Imports
import numpy as np
import matplotlib.pyplot as plt          # For all our plotting needs
from sklearn.model_selection import train_test_split

##### GENERATING SYNTHETIC DATA #####
##### Given #####
# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

##### Given #####
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

##### Given #####
plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')
# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')
plt.show()

##### Given #####
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set
# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
plt.show()

##### 1. REGRESSION WITH POLYNOMIAL BASIS FUNCTIONS (30 POINTS) #####
##### a. axn180041 #####
# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
    out_mat = []
    for i in range(len(X)):
        in_mat = []
        for j in range(d):
            in_mat.append(X[i]**j)
        out_mat.append(in_mat)
        vander_mat = np.flip(np.array(out_mat),axis=1)
        #print (vander_mat)
    return vander_mat

##### b. axn180041 #####
# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    w = np.linalg.inv((Phi.T)@Phi)@(Phi.T)@y
    return w

##### c. axn180041 #####
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    ylen = len(y)
    total = np.sum(np.power(np.subtract(y,Phi@w),2))
    mse = total/ylen
    return mse

##### d. Given #####
w = {}  # Dictionary to store all the trained models
validationErr = {}  # Validation error of the models
testErr = {}  # Test error of all the models
for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)  # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)  # Learn model on training data

    Phi_val = polynomial_transform(X_val, d)  # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data

    Phi_tst = polynomial_transform(X_tst, d)  # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])
#####
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)
plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])
plt.show()

##### Discussion: axn180041 #####
#Goal of using Linear regression algorithm is to find the best fit line using Least Square method.
# We are calculating Mean squared error and from the graph if MSE is 1, it is correct to forecast next value.
#we have plotted error against polynomial degree.
#we calculated 'w' which is considered coefficient to reduce error
#For all n training points, we calculate the difference between estimated values and original values
#Highest MSE is considered better and in the graph when degree is 18, error is min

##### 2. REGRESSION WITH RADIAL BASIS FUNCTIONS[70 POINTS] #####
##### a. axn180041 #####
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    out_mat=[]
    for i in range(len(X)):
        in_mat=[]
        for j in range(len(B)):
            squaring=(X[i]-B[j])**2
            in_mat.append(-gamma*(squaring))
        out_mat.append(in_mat)
    return np.exp(out_mat)

##### b. axn180041 #####
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
    a=np.dot(Phi.transpose(),Phi)
    b=(lam*np.identity(len(Phi.transpose())))
    c=np.add(a,b)
    final=np.linalg.inv(c)
    return np.dot(final,np.dot(Phi.transpose(),y))

##### c. axn180041 #####
w={}
validationErr={}
testErr={}
lam=((10)**(-3))
while lam<=((10)**(3)):
    Phi_trn = radial_basis_transform(X_trn,X_trn)
    w[lam] = train_ridge_model(Phi_trn, y_trn,lam)
    Phi_val = radial_basis_transform(X_val,X_trn)
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam])
    Phi_tst = radial_basis_transform(X_tst, X_trn)
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam])
    lam=lam*10
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.xscale("log")
plt.show()
##### d. axn180041 #####
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
lam=((10)**(-3))
list2=list()
while lam<=((10)**(3)):
    list2.append(lam)
    X_lam = radial_basis_transform(x_true, X_trn)
    y_lam = X_lam @ w[lam]
    plt.plot(x_true, y_lam, marker='None', linewidth=2)
    lam=lam*10
plt.legend(['true'] + list2)
plt.axis([-8, 8, -15, 15])
plt.show()

##### d. axn180041- Discussion #####
#Our goal is to determine how the linearity of the model changes with Lambda
#When we don't have more training samples, regularization is used.
# So we added weight Lambda and added to previous polynomial.
#We can observe that minimum error is at Lambda =10^-3
# As we increase lambda, linearity increases.
#Lambda should be minimum to avoid overfitting.
