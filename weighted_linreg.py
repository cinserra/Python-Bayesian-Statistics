from pylab import *      # import matplotlib
from mpl_toolkits.axes_grid.axislines import * # import toolkits to get minor ticks in plot
from scipy import stats
import numpy as np
import linmix, random #you need to downlaod linmix to use this script

# igure/plot settings
rc('text', usetex=True)
rc('font',family='Times New Roman')
rc('xtick', labelsize=13)
rc('ytick', labelsize=13)
fig = plt.figure(figsize=(7,6))

#let's define some functions to reprodce the confidence boundaries
#the first function is a first order polynomial
def func(x, a, b):
    '''linear 2-param function.'''
    return a + (b * x)

#this define the confidence regions
def predband(x, xd, yd, f_vars, conf=0.95):
    """
    Code adapted from Rodrigo Nemmen's post:
    http://astropython.blogspot.com.ar/2011/12/calculating-prediction-band-
    of-linear.html

    Calculates the prediction band of the regression model at the
    desired confidence level.

    Clarification of the difference between confidence and prediction bands:
    The 95%
    prediction band is the area in which you expect 95% of all data points
    to fall. In contrast, the 95% confidence band is the area that has a
    95% chance of containing the true regression line."

    References:
    1. http://www.JerryDallal.com/LHSP/slr.htm, Introduction to Simple Linear
    Regression, Gerard E. Dallal, Ph.D.
    """

    alpha = 1. - conf    # Significance
    N = xd.size          # data sample size
    var_n = len(f_vars)  # Number of variables used by the fitted function.

    # Quantile of Student's t distribution for p=(1 - alpha/2)
    q = stats.t.ppf(1. - alpha / 2., N - var_n)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *f_vars)) ** 2))

    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)

    # Predicted values (best-fit model)
    yp = func(x, *f_vars)
    # Prediction band
    dy = q * se * np.sqrt(1. + (1. / N) + (sx / sxd))

    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy

    return lpb, upb


ax=fig.add_subplot(111)

#let's produce some random data and errors
X = random.sample(range(1, 100), 40)
Y = random.sample(range(1, 100), 40)
#this is a trick to have float errors spanning from the order of 10^2 to 10^-2
X_err =  (1.*np.array(random.sample(range(1, 100), 40))) / (1.*np.array(random.sample(range(1, 100), 40)))
Y_err =  (1.*np.array(random.sample(range(1, 100), 40))) / (1.*np.array(random.sample(range(1, 100), 40)))

# creates array since linmix works with arrays and not lists
Xfa = np.array(X)
Yfa = np.array(Y)
Xfa_err = np.array(X_err)
Yfa_err = np.array(Y_err)
#run a Monte carlo Markov Chain producing at least 5000 iterations (a.k.a. 5000 different
# linear regressions)
lm = linmix.LinMix(Xfa, Yfa, xsig=Xfa_err, ysig=Yfa_err, K=3)
lm.run_mcmc(miniter=5000, maxiter=100000,silent=True)

# print the average parameters several statistical
print ''
print " Pearson test ", stats.pearsonr(X, Y)
print "Spearman test ", stats.spearmanr(X, Y)
print ''
print '----- Bayesian linear regression with error in both X and Y -----'
print 'Beta = ',lm.chain['alpha'].mean(),'+/-', lm.chain['alpha'].std()
print 'Alpha = ',lm.chain['beta'].mean(), '+/-',lm.chain['beta'].std()
print 'Sigma = ',np.sqrt(lm.chain['sigsqr'].mean()), '+/-', np.sqrt(lm.chain['sigsqr'].std())
print 'Variance = ',lm.chain['sigsqr'].mean(), '+/-', lm.chain['sigsqr'].std()
print 'Correlation = ',lm.chain['corr'].mean(), '+/-', lm.chain['corr'].std()

# intialise value (see below)
value = 0
for i in xrange(0, len(lm.chain), 25):
    xs = np.arange(Xfa.min(),Xfa.max(),0.01)
    if lm.chain['beta'][i] < 0.01:
        value = +1
ys = lm.chain['alpha'].mean() + xs * lm.chain['beta'].mean()

#check how many times do you have a zero slope (for statistical cenrtainty)
print ''
print 'Number of times alpha is zero:', value, 'over ', len(lm.chain), 'iterations'
print ''

# plot the average linear regression and 2sigma boundaries = 95% f the cases will be in the confidence
# region (conf=0.95)
sigma = np.sqrt(lm.chain['sigsqr'].mean())
plot(xs, ys,ls='--',color='#1f77b4',label='_nolegend_')
popt = (lm.chain['alpha'].mean(), lm.chain['beta'].mean())
low_l, up_l = predband(xs, Xfa, Yfa, popt, conf=0.95) #it creates the boundary using the functions defined above
ax.fill_between(xs,up_l,low_l,where=None,alpha=0.2,facecolor='#1f77b4',edgecolor ='#1f77b4',zorder=-3, label='2$\sigma$')
plot(xs, up_l,ls='-',color='#1f77b4',label='_nolegend_')
plot(xs, low_l,ls='-',color='#1f77b4',label='_nolegend_')

#plot data, errors and custom legend
errorbar(X,Y,xerr=X_err,yerr=Y_err,marker='.',ms=8,color='grey',ecolor='grey',ls='None',label='_nolegend_')
plot(X,Y,marker='*',ms=12,color='k',ls='None',zorder=10,label='Data')
plt.legend(loc=3,prop={'size':11})

xlabel('X',fontsize=20)
ylabel('Y',fontsize=20)

ax.minorticks_on()
show()
