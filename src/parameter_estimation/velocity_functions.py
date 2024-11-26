import numpy as np

def diehl(thetas, xdata):
    # theta0=V0
    # theta1=X¯
    #thetas2=q
    return thetas[0] / (1 + ((xdata / thetas[1]) ** thetas[2]))

def cole(thetas, xdata):
    # theta0=k
    # theta1=n
    return thetas[0] * (xdata ** (-thetas[1]))

def takacs(thetas, xdata):
    # thetas0=vo
    # thetas1=rh
    # thetas2=rp
    return thetas[0] * (np.exp(-thetas[1] * xdata) - np.exp(-thetas[2] * xdata))

def vesilind(thetas, xdata):
    # theta0=v0
    # theta1=rv
    return thetas[0] * np.exp(-thetas[1] * xdata)