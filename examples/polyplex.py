
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[12]:


from rankpruning.generate_noise_matrix import generate_noise_matrix_from_trace, noise_matrix_is_valid


# In[3]:


from __future__ import print_function
import numpy as np
np.set_printoptions(linewidth=200)
colors = [plt.cm.tab10(i) for i in range(plt.cm.tab10.N)] + [plt.cm.Vega10(i) for i in range(plt.cm.Vega10.N)]


# In[4]:


# Plotting functions and imports
from matplotlib import pyplot as plt

# For pretty figure plotting
import seaborn as sns
sns.set(style='white', font_scale=3)

# Important! Make fonts Type I fonts (necessary for publishing in ICML and other conference)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
# plt.rcParams.update(params)

plt.rc('text',usetex=True)


# In[5]:


def slope_intercept(point1, point2):
    '''Returns the slope and intercept between point1 and point2.
    
    Parameters
    ----------    
    point1 : tuple
       e.g. (1.3, 4)
       
    point2 : tuple
    
    Output
    ------      
      A tuple(slope, intercept)'''
    
    (x0, y0) = point1
    (x1, y1) = point2
    slope = (y1 - y0) / float(x1 - x0)
    intercept = y1 - slope * x1
    return slope, intercept

def joint_bounds(py):
    '''Computes three lists: noise_matrix_trace, joint_trace_min, joint_trace_max that when
    plotted, the noise_matrix_trace values represent x-values and the joint_trace_min and 
    joint_trace_max values represent the y-value min and maximium ranges. Together, these 
    three lists fully characterize the polyplex.

    Parameters
    ----------

    py : np.array (shape (K, 1))
        The fraction (prior probability) of each true, hidden class label, P(y = k)
        
    Output
    ------
        A tuple of lists of floats (noise_matrix_trace, joint_trace_min, joint_trace_max)
        each of length K+1, where K = len(py). When plotted, the noise_matrix_trace values
        represent x-values and the joint_trace_min and joint_trace_max values represent the
        y-value min and maximium ranges. These three lists fully characterize the polyplex.'''
    
    K = len(py)
    py = np.sort(py)[::-1] 
    noise_matrix_trace = np.arange(K+1)    
    joint_trace_min, total = [1], 1
    for p in py:
        joint_trace_min.append(total - p)
        total -= p 
    joint_trace_min = np.flip(joint_trace_min, axis=0)
    joint_trace_max = 1 - joint_trace_min[::-1]
    return noise_matrix_trace, joint_trace_min, joint_trace_max

def joint_min_max(noise_matrix_trace, py):
    '''Computes the min and max bounds on the trace(P_{s,y}), the trace of the 
    joint distribution, given the trace of the noise matrix and p(y).

    Parameters
    ----------

    py : np.array (shape (K, 1))
        The fraction (prior probability) of each true, hidden class label, P(y = k)
        
    Output
    ------
        A tuple of two floats (y_min, y_max) representing the bounds on the trace of the joint.'''
    _, y_mins, y_maxs = joint_bounds(py)
    if int(noise_matrix_trace) == noise_matrix_trace:
        return y_mins[int(noise_matrix_trace)], y_maxs[int(noise_matrix_trace)]
    else:
        slope_min = y_mins[int(noise_matrix_trace)+1] - y_mins[int(noise_matrix_trace)]
        y_min = (noise_matrix_trace - int(noise_matrix_trace)) * slope_min + y_mins[int(noise_matrix_trace)]
        slope_max = y_maxs[int(noise_matrix_trace)+1] - y_maxs[int(noise_matrix_trace)]
        y_max = (noise_matrix_trace - int(noise_matrix_trace)) * slope_max + y_maxs[int(noise_matrix_trace)]
        return y_min, y_max


# In[6]:


def draw_polyplex(py, intersecting_lines=False):
    py = [round(p, 3) for p in py]
    plt.figure(figsize=(10, 7))
    K = len(py)
    x, y_min, y_max = joint_bounds(py)
    min_slopes = [round(slope_intercept(*zip(x, y_min)[i-1:i+1])[0], 4) for i in range(1, len(x))]
    max_slopes = [round(slope_intercept(*zip(x, y_max)[i-1:i+1])[0], 4) for i in range(1, len(x))]
    _ = plt.plot(x[:2], y_min[:2], label='min', color='black', linestyle='--')    
    _ = plt.plot(x[:2], y_max[:2], label='max', color='black', linestyle='--')
    _ = plt.plot(x[1:], y_min[1:], label='min', color='black')    
    _ = plt.plot(x[1:], y_max[1:], label='max', color='black')
    _ = plt.vlines(1, y_min[1], y_max[1], color='black')
    # Annotate with slopes
    for i in range(len(x)-1):
        _ = plt.text(x[i]+.5, (y_min[i]+y_min[i+1])/2, str(min_slopes[i]))
        _ = plt.text(x[i]+.5, (y_max[i]+y_max[i+1])/2, str(max_slopes[i]))
    ax = plt.gca()
    ax.fill_between(x[1:], y_min[1:], y_max[1:], color='gray', alpha=0.25)
    if intersecting_lines:
        for i, trace in enumerate(range(K+1)):
            _ = plt.vlines(trace, 0, 1, color=colors[i], linestyle=':')
            _ = plt.hlines(joint_min_max(trace, py),0,K, color=colors[i], linestyle='--')
#     _ = plt.suptitle('Polyplex of joint distribution $P_{s,y}$')
    _ = plt.title('$p(y) =$' + str(py), fontsize = 25)
    _ = plt.xlabel(r'$trace(P_{s|y})$') 
    _ = plt.ylabel(r'$trace(P_{s,y})$')
    _ = plt.xticks([0, K//2, K] if K%2==0 else [0, K//2, K//2 + 1,K]) 
    return ax


# In[20]:


savefig = False

pys = [[.01,.39,.6], [0.2]*5, [.1,.3,.6], [.3,.3,.4],[.2,.6,.2],[.1,.2,.7],[.1,.15,.15,.6],[.1,.1,.2,.6],[1./3,1./3,1./3], [.02,.05,.08,.15,.15,.25,.3], [.05,.1,.15,.35,0.35] ]
pys = pys + [np.arange(1,k+1, dtype=float) / sum(range(k+1)) for k in range(2,14)]
for py in pys:
    ax = draw_polyplex(py)
    
    K=len(py)
    for nm_avg_trace in np.arange(0.01, 1.01, 0.03):
#         py = np.arange(1,K+1, dtype=float) / sum(range(K+1))
        nm = generate_noise_matrix_from_trace(K, nm_avg_trace, valid_noise_matrix=False, py=py)
        valid = noise_matrix_is_valid(nm, py)
        joint_trace = np.trace(nm*py)
#         print('ps is', (nm*py).sum(axis=1))
        _ = ax.text(nm_avg_trace*K, joint_trace, s='v' if valid else 'n', color='red')
    
    
#     for z in np.arange(100):
# #         py = np.arange(1,K+1, dtype=float) / sum(range(K+1))
#         nm = generate_noise_matrix_from_trace(K, nm_avg_trace, valid_noise_matrix=True, py=py)
#         joint_trace = np.trace(nm*py)
# #         print('ps is', (nm*py).sum(axis=1))
#         _ = ax.plot([nm_avg_trace*K], [joint_trace], marker='o', color='red')
    
    if savefig:
        plt.savefig('figs/polyplices/polyplex_'+str(py)+'_withvs.pdf', pad_inches=0.0, bbox_inches='tight')
        
        
    _ = draw_polyplex(py, intersecting_lines=True)
    if savefig:
        plt.savefig('figs/polyplices/polyplex_'+str(py)+'_withlines.pdf', pad_inches=0.0, bbox_inches='tight')
    
    _ = draw_polyplex(py, intersecting_lines=False)
    if savefig:
        plt.savefig('figs/polyplices/polyplex_'+str(py)+'_plain.pdf', pad_inches=0.0, bbox_inches='tight')    


# ### Experiments and empirical estimates.

# In[252]:


ntrials = 20000
pys = [[.01,.39,.6], [.1,.3,.6], [.3,.3,.4],[.2,.2,.6],[.1,.2,.7],[.1,.15,.15,.6],[.1,.1,.2,.6],[1./3,1./3,1./3] ]
pys = [np.array(i) for i in pys]
avg_traces = [1.2/3, 0.5, 2.8/3]
for py in pys:
    K = len(py)
    for tr in avg_traces:
        result = [np.sum(generate_noise_matrix_from_trace(K, tr, py=py).diagonal() * py) for i in range(ntrials)]
        print("py = ", py, "Avg trace = ", np.round(tr, 3), "Trace = ", tr*K, "  |  ", joint_min_max(tr*K, py)[0].round(2), np.min(result).round(3), np.max(result).round(3),  joint_min_max(tr*K, py)[1].round(2))
    print()   


# ### Observations
# 1. When avg_trace = 0.5, max_trace_joint + min_trace_joint = 1 and avg_trace = mean(min, max)
# 2. max and min of joint trace spread farther apart as py entropy/variance increases
# 3. max and min of joint trace increase as avg_trace increases
# 4. avg_trace = min = max when py is uniform

# In[501]:


py = np.array([.1,.3,.6])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.1, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[502]:


py = np.array([.2,.2,.6])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.1, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[504]:


py = np.array([.01,.19,.8])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.1, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[505]:


py = np.array([.1,.15, .15,.6])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.1, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[13]:


py = np.array([.6, .1, .1, .2])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.3, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[88]:


py = np.array([.1,.1,.2,.6])
joint_bounds(py)
x, y1, y2 = joint_bounds(py)
plt.plot(x[1:], y1[1:], label='max')
plt.plot(x[1:], y2[1:], label='min')
plt.legend(title='joint trace')


# In[506]:


py = np.array([.1,.1,.2,.6])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.1, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[10]:


py = np.array([.02,.05,.08,.15,.15,.25,.3])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.8, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    print(trace, end=", ")
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)
print()

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
# max_intercept_before = 
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[89]:


x, y1, y2 = joint_bounds([.02,.05,.08,.15,.15,.25,.3])
plt.plot(x[1:], y1[1:], label='max')
plt.plot(x[1:], y2[1:], label='min')
plt.legend(title='joint trace')


# In[564]:


[.05,.1,.15,.35,0.35]

x = np.arange(2,K+1)
y = np.array([max_data[abs(trace_values-k)<1e-3][0] for k in x])
x
y
for i in range(1, len(x)):
    point1 = x[i-1], y[i-1]
    point2 = x[i], y[i]
    linear_params(point1, point2)
    
    
x = np.arange(2,K+1)
y = np.array([min_data[abs(trace_values-k)<1e-3][0] for k in x])
x
y
for i in range(1, len(x)):
    point1 = x[i-1], y[i-1]
    point2 = x[i], y[i]
    linear_params(point1, point2)


# In[562]:


py = np.array([.05,.1,.15,.35,0.35])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.3, K +.01, 0.1)
ntrials = 50000
for trace in trace_values:
    print(trace, end=", ")
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)
print()

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
# max_intercept_before = 
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[90]:


x, y1, y2 = joint_bounds([.05,.1,.15,.35,0.35])
plt.plot(x[1:], y1[1:], label='max')
plt.plot(x[1:], y2[1:], label='min')
plt.legend(title='joint trace')


# In[529]:


py = np.array([.05,.2,.2,.2,0.35])
K = len(py)
min_data = [] 
max_data = []
trace_values = np.arange(1.3, K +.01, 0.1)
ntrials = 5000
for trace in trace_values:
    print(trace, end=", ")
    trials = [np.sum(generate_noise_matrix_from_trace(K, trace/float(K), py=py).diagonal() * py) for i in range(ntrials)]
    _ = min_data.append(min(trials))
    _ = max_data.append(max(trials))
min_data = np.array(min_data)
max_data = np.array(max_data)
print()

# Slopes
max_k_minus_1_val = max_data[abs(trace_values-(K-1))<1e-3]
min_k_minus_1_val = min_data[abs(trace_values-(K-1))<1e-3]
delta_x = (K-1) - min(trace_values)
max_slope_before = (max_k_minus_1_val - min(max_data)) / delta_x
# max_intercept_before = 
min_slope_before = (min_k_minus_1_val - min(min_data)) / delta_x
max_slope_after = 1 - max_k_minus_1_val
min_slope_after = 1 - min_k_minus_1_val

print('max slope before trace =', K-1, 'is', np.round(max_slope_before, 2))
print('min slope before trace =', K-1, 'is', np.round(min_slope_before, 2))
print('max slope after trace =', K-1, 'is', np.round(max_slope_after, 2))
print('min slope after trace =', K-1, 'is', np.round(min_slope_after, 2))

_ = plt.plot(trace_values, max_data, label='max')
_ = plt.plot(trace_values, min_data, label='min')
_ = plt.legend()
_ = plt.xlabel('trace')
_ = plt.title(str(py))


# In[285]:


py = np.array([0.1, 0.3, 0.6])
trace = 1.1


# In[361]:


P(s|y) * p(y) = p(s,y)


# In[180]:


py = np.array([0.1,0.25,0.3, 0.35])
K = len(py)
trace = 1.1
low, high = joint_min_max(trace, py)
low, high
joint_trace = np.random.uniform(low=low, high=high)
# joint_trace = 0.27
results = []
for i in range(500000):
    valid_probs = False
    while not valid_probs:
        joint_diagonal = generate_n_rand_probabilities_that_sum_to_m(K, joint_trace)
        valid_probs = (joint_diagonal < py).all()
    results.append(joint_diagonal)
results = np.array(results)

_ = plt.hist(results[:,0], alpha=.5, bins = 50)
_ = plt.hist(results[:,1], alpha=.5, bins = 50)
_ = plt.hist(results[:,2], alpha=.5, bins = 50)
_ = plt.hist(results[:,3], alpha=.5, bins = 50)
for k in range(0,K):
    _ = plt.hist(results[:,k], alpha=.5, bins=50)


# In[377]:


py = [1./10]*10
K = len(py)
trace = 1.3
get_ipython().run_line_magic('time', 'generate_noise_matrix_from_trace(K, trace/K, py=py)')
# %time gen_noise_matrix_fast(trace, py)


# In[68]:


def gen_noise_matrix_fast(trace, py):
    K = len(py)
    low, high = joint_min_max(trace, py)
 
    # results0 = []
    # results = []
    valid = False
    while not valid:
        joint_trace = np.random.uniform(low=low, high=high)
        # Generate joint p(s,y) diagonal such that no diagonal term 
        #   is greater than its corresponding p(y)
        valid_wrt_py = False
        while not valid_wrt_py:
            joint_diagonal = generate_n_rand_probabilities_that_sum_to_m(K, joint_trace)
            joint_diagonal *= trace / np.sum(joint_diagonal / py)
            valid_wrt_py = (joint_diagonal <= py).all()
 
    #     noise_matrix_diagonal = joint_diagonal / py
        # 'trace of noise matrix', sum(noise_matrix_diagonal)
        # 'joint diagonal before calibration', joint_diagonal
        # 'trace of joint before calibration', sum(joint_diagonal)
    #     joint_diagonal *= trace / np.sum(joint_diagonal / py)
        joint_trace = np.sum(joint_diagonal)
        results.append(joint_trace)
        # 'trace of joint diagaonal after calibration', joint_trace
    #     print(joint_trace)
    #     assert(joint_trace >= low and joint_trace <= high)
    #     print(joint_trace)
 
        joint_matrix = np.empty(shape=(K, K))
        for col in range(K):
    #         print('py[col]-joint_diagonal[col]', py[col]-joint_diagonal[col], py, joint_diagonal)
            noise_rates_col = list(generate_n_rand_probabilities_that_sum_to_m(
                n=K-1, 
                m=py[col]-joint_diagonal[col],
            ))
            for row in range(K):
                if row == col:
                    joint_matrix[row][col] = joint_diagonal[col]
                else:
                    joint_matrix[row][col] = noise_rates_col.pop()
 
        noise_matrix = joint_matrix / py
    #     noise_matrix
    #     results0.append(noise_matrix)
        valid = noise_matrix_is_valid(noise_matrix, py)
#         joint_matrix
    #     'noise_matrix trace', np.trace(noise_matrix)
    #     'joint trace', np.trace(joint_matrix)
    #     (joint_matrix).sum(axis=0), py
    #     (joint_matrix).sum(axis=1)
    #     (noise_matrix).sum(axis=0)
     
    return noise_matrix


# In[335]:


ntrials = 20
pys = [[.01,.39,.6], [.1,.3,.6], [.3,.3,.4],[.2,.2,.6],[.1,.2,.7],[.1,.15,.15,.6],[.1,.1,.2,.6],[1./3,1./3,1./3] ]
pys = [np.array(i) for i in pys]
avg_traces = [1.01/3, 1.2/3, 0.5, 2.8/3]
for py in pys:
    K = len(py)
    for tr in avg_traces:
        print("py = ", py, "Avg trace = ", np.round(tr, 3), "Trace = ", tr*K)        
        get_ipython().run_line_magic('time', 'result = [np.sum(generate_noise_matrix_from_trace(K, tr, py=py).diagonal() * py) for i in range(ntrials)]')
        get_ipython().run_line_magic('time', 'result0 = [np.sum(gen_noise_matrix_fast(tr*K, py=py).diagonal() * py) for i in range(ntrials)]')
#         print("py = ", py, "Avg trace = ", np.round(tr, 3), "Trace = ", tr*K, "  |  ", joint_min_max(tr*K, py)[0].round(2), np.min(result).round(3), np.max(result).round(3),  joint_min_max(tr*K, py)[1].round(2))
    print()   


# In[325]:


nm = gen_noise_matrix_fast(trace=1.8, py=py)


# In[ ]:


ps[i] * py[i] < joint_noise[i][i]


# ### questions
# 1. can we make claims about the joint matrix trace in terms of learnability?
# 2. Can we enforce ps in joint matrix counts in rank pruning?
