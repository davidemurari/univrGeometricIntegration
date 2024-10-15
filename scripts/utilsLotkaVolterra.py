import numpy as np

#Parameters of the model
alpha = 2.
beta = 1.
gamma = 1.
delta = 1.

x0,y0 = 0.1,1.9
z0 = np.array([x0,y0])

def Gonzalez(V,gradV,x,y):
   tol = 1e-10
   if np.linalg.norm(y-x,ord=2)<tol:
        return gradV(x)
   else:
        return gradV(avg(x,y))+(V(y)-V(x)-np.dot(gradV(avg(x,y)),y-x))*(y-x)/np.linalg.norm(y-x,ord=2)**2

def ItohAbe(V,gradV,x,y):
  #This method returns the discrete gradient of V evaluated at x and y
  #The chosen discrete gradient is the coordinate increment one
  res = np.zeros_like(x)
  for i in range(len(x)):
    if (y[i]!=x[i]):
      res[i] = V(np.concatenate((y[:i+1],x[i+1:]))) - V(np.concatenate((y[:i],x[i:])))
      res[i] /= (y[i]-x[i])
    else:
      res[i] = gradV(x)[i]
  return res

def H(z):
    x,y = z[0],z[1]
    return delta*x-gamma*np.log(x)+beta*y-alpha*np.log(y)

def gradH(z):
    x,y = z[0],z[1]
    return np.array([
        delta-gamma/x,
        beta-alpha/y
    ])

def avg(z,zh):
    return (z+zh)/2

def discGrad(z,zh):
    return ItohAbe(H,gradH,z,zh)

def S(z):
    x,y = z[0],z[1]
    return np.array([[0,-x*y],[x*y,0.]])

def discS(z,zh):
    return S(avg(z,zh))
