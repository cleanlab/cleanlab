
# coding: utf-8

# Copyright (c) 2017-2050 Curtis G. Northcutt
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


# ## The Polyplex
# 
# ### The polyplex is a geometric solution for the domain of $trace (P_{s,y})$ given $trace (P_{s \vert y})$ for some the latent prior $p(y)$. Like the simplex, the polyplex describes the domain of a probabilistic quantity, but unlike the simplex, the coordinates need not sum to one and the shape is instead defined by a convex polyhedron, hence the name. Understanding the domain of the trace, as opposed to the matrices themselves, is of fundamental importance to confident learning because the diagonal terms of these matrices are what determine learnability and the class re-weighting coefficients. Polyplices connect our learnability theory with our algorithms. Consider the canonical example when $trace (P_{s \vert y}) = m$, then $P_{s \vert y}=\mathbf{I}$ and all noise rates (non-diagonal entries) are zero. Thus, $P_{s,y} = \mathbf{I}$ and $trace (P_{s,y}) = m$. In this vacuous example, $s = y, p(s) = p(y), P_{s,y} = \mathbf{I} \cdot p(s)$, and $P_{y \vert s} = P_{s \vert y} = \mathbf{I}$, all of which was determined from the trace. The role of the polyplex is to generalize to non-vacuous cases by using the geometry of confident learning to solve for $trace (P_{s,y})$ given $trace (P_{s \vert y})$ for some $p(y)$.

from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
import numpy as np

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
    noise_matrix_trace : float
        The sum of the diagonals of the noise matrix P(s = k' | y = k)
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
