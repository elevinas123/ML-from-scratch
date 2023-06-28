from array import array

import numpy as np
import matplotlib.pyplot as plt

def mse(y, yPred):
  return ((y-yPred)**2)/2
def sparseEntropy(y, yPred):
  sum = 0
  for n, i in enumerate(yPred):
    if i==0:
      i=0.0001
    sum+=y[n]*np.log(i)
  return sum*-1
class model():
  def __init__(self, batchSize, inp, outp, numLayers, hidden):
    self.inp = inp
    self.outp = outp
    self.hidden = hidden
    self.t = 1
    self.x = False
    self.w = []
    self.z = []
    self.a = []
    self.b = []
    self.batch_size = batchSize
    self.beta = 0.9
    self.alfa = 0.9
    self.Dense(inp, hidden)
    for i in range(numLayers):
      self.Dense(hidden, hidden)
    self.Dense(hidden, outp)
    self.lamda = 0.001
    self.lr = 0.001
  def loop(self, epochs, x, y):
    
    y = y[0]
    l =0
    for epoch in range(epochs):
      
        sum = 0
        sumD = 0
        for n, i in enumerate(x):
          pr = self.forward(i)[0]
          sum+=mse(np.sin(i), pr)
          sumD+=i-np.sin(i)
        self.z.insert(0, self.x/n)
        self.back(sumD/n)
        loss = sum
        self.x = False
        if epoch % 10  ==0:
          print("epoch:", epoch, " loss:", loss)
        
      
    return
  def softmax(self, x):
  
    e_x = np.exp(x[0] - np.max(x[0]))
    return e_x / e_x.sum(axis=0) 

  def forward(self, z):
    z = np.reshape(z, (1, self.inp))
    if self.x==False:
      self.x = z
    else:
      self.x+=z
    for n, i in enumerate(self.w[:-1]):
      a  = z@i+self.b[n]
      self.a.append(a)
      z = self.ReLU(a)
      self.z.append(z)
    self.o = z@self.w[-1]+self.b[-1]
    
    return self.o[0]
  
  def ReLU(self, X, list=True):
    if list==True:
      q = []
      for n, i in enumerate(X):
        q.append([])
        for j in i:
          if j<0:
            j = 0
          q[n].append(j)
    else:
      if X>0:
        q=X
      else:
        q=0
    return np.array(q)
  def d_ReLU(self, x):
    if x>0:
      q=1
    else:
      q=0
    return q
  def Dense(self, inp, hidden):
      self.w.append(np.random.uniform(-1, 1, (inp, hidden)))
      self.b.append(np.random.uniform(0, 0, (1, hidden)))
  def d_entropy(self, o, y):
    return o-y
  def back(self, loss):
    aT = self.a[::-1]
    zT = self.z[::-1]
    w = self.w[:]
    b = self.b[::-1]
    wNew = []
    bNew = []
    a = [[] for i in range(len(w)+1)]
    a[0] = [loss for i in self.w[-1]]
    for n, i in enumerate(reversed(w)):
      wNew.append([])
      bNew.append([])
      for n1, j in enumerate(i):
        wNew[n].append([])
        sum=0
        for n2, k in enumerate(j):
          g = (a[n][n2]*zT[n][0][n1])
          if g>1:
            g=1
          elif g<-1:
            g = -1
          wNew[n][n1].append(k - g*self.lr)
          if n!=len(w)-1:
            sum+=a[n][n2]*k*self.d_ReLU(aT[n][0][n1])
        a[n+1].append(sum)
      c = np.squeeze(b[n])
      if np.size(c)==1:
        c = np.array([c])
      for n1, j in enumerate(c):
        g = a[n][n1]
        if g>5:
            g=5
        elif g<-5:
          g = -5
        bNew[n].append(j-g*self.lr)
    self.w = wNew[::-1]
    self.b = bNew[::-1]
    self.a =[]
    self.z =[]
    return
t = np.random.uniform(-6, 6, 1000)
model = model(1, 1, 1, 2, 3)
model.loop(1000, t, [np.sin(i) for i in t])
print("done")
t = np.linspace(0, 2*np.pi, 100)  # replace with your actual t values
data1 = [np.sin(i) for i in t]
data2 = [model.forward(i) for i in t]  # assuming model.forward() is a function that takes a single argument

plt.figure(figsize=(10, 6))

# Plot data1
plt.plot(t, data1, label='np.sin(i)', color='blue')

# Plot data2
plt.plot(t, data2, label='model.forward(i)', color='red')

plt.title('Plot of np.sin(i) and model.forward(i)')
plt.xlabel('t')
plt.ylabel('Value')
plt.legend(loc='best')  # Display labels in the best location

plt.show()