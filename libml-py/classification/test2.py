import matplotlib.pyplot as plt
import numpy as np
import random
import logistic_regression as logreg
np.random.seed(78)
def sample_binary_data():
    m_0 = []
    m_1 = []
    mu_0 = [1,0]
    mu_1 = [0,1]
    cov = [[1,0],[0,1]]

    for _ in xrange(10):
        x, y = np.random.multivariate_normal(mu_0,cov,1).T
        m_0.append([x[0],y[0]])
        x, y = np.random.multivariate_normal(mu_1,cov,1).T
        m_1.append([x[0],y[0]])

    x_0, y_0 = [], []
    x_1, y_1 = [], []
    cov = [[1./5,0],[0,1./5]]

    for _ in xrange(100):
        m_k = random.choice(m_0)
        x, y =  np.random.multivariate_normal(m_k,cov,1).T
        x_0.append(x[0])
        y_0.append(y[0])
        
        m_k = random.choice(m_1)
        x, y =  np.random.multivariate_normal(m_k,cov,1).T
        x_1.append(x[0])
        y_1.append(y[0])

    return (x_0,y_0),(x_1,y_1)

def plot_data(data0, data1, boundary=False, Theta=None, X=None, y=None):
    assert len(data0) == len(data1) == 2
    x_0, y_0 = data0
    x_1, y_1 = data1
    data_0 = plt.plot(x_0,y_0,'o',color='blue',fillstyle='none',label='Data 0')
    data_1 = plt.plot(x_1,y_1,'o',color='orange',fillstyle='none',label='Data 1')
    if boundary:
        if np.size(X,1) <= 2:
            plot_x = [min(X[:,1])-2, max(X[:,1])+2]
            slope = -1./Theta[2]*Theta[1]
            plot_y = [slope*plot_x[0] + Theta[0], slope*plot_x[1] + Theta[0]]
            plt.plot(plot_x,plot_y,color='r',label='Decision Boundary',scalex=False,scaley=False)
    plt.legend(loc=0)
    plt.show()
    return plt

(x_0,y_0), (x_1,y_1) = sample_binary_data()
xdata = []
ydata = []
for i in xrange(len(x_0)):
    xdata.append([x_0[i],y_0[i]])
    ydata.append([0])
    xdata.append([x_1[i],y_1[i]])
    ydata.append([1])
new_data = zip(xdata,ydata)
xdata, ydata = zip(*sorted(new_data, key=lambda x: x[0][0]))
X = np.array(xdata)
y = np.array(ydata)

model = logreg.LogisticRegression()
model.fit(X,y,normalize_data=True)
print model.theta
# data0, data1 = sample_binary_data()
# plt = plot_data((x_0,y_0),(x_1,y_1), boundary=True, Theta=model.theta, X=X, y=y)
