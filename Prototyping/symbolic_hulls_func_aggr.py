# Math and geometry libraries
import numpy as np
import sympy as sp
from sympy.polys.polytools import primitive
from scipy.spatial import ConvexHull

def gradient(f, vars):
    '''
    Computes the gradient of a function f with respect to the variables vars.

    Parameters:
        f (sympy expression): The function to take the gradient of.
        vars (sympy symbols): The variables to take the gradient with respect to.
    
    Returns:
        gradient (tuple): The gradient of the function.
    '''
    return tuple(f.diff(var) for var in vars)

def td_legendre_transform(x, p, f):
    '''
    Takes the Legendre transformation of a function f with respect to the variables x and the conjugate p.

    Parameters:
        x (sympy symbols): The variables of the function.
        p (sympy symbols): The conjugate variables of the function.
        f (sympy expression): The function to take the Legendre transformation of.

    Returns:
        f_transformed (sympy expression): The transformed function.
    '''
    f_transformed = f.copy()
    for i in range(len(x)):
        f_transformed -= p[i]*x[i]

    return f_transformed

def recursive_discriminant(expr, vars):
    '''
    Takes the discriminant over all variables in a given expression.

    Parameters:
        expr (sympy expression): The expression to take the discriminant of.

    Returns:
        discriminant (sympy expression): The discriminant of the expression.
    '''
    vars_list = list(vars)
    def recursive_discriminant_helper(expr, vars):
        # if there are no variables, return the expression
        if vars == []:
            return expr
        
        for var in vars_list:
            if expr.has(var):
                vars_list.remove(var)
                return sp.discriminant(recursive_discriminant(expr, vars_list), var)
            
    return recursive_discriminant_helper(expr, vars_list)

def hpboundry_f2_to_f1(f1, f2):
    '''
    Uses the Legendre transformation to find the boundary between two functions in the projected space.

    Parameters:
        f1 (sympy expression): The first function.
        f2 (sympy expression): The second function.
    
    Returns:
        discriminant (sympy expression): the polynomial equation that defines the boundary between the two functions (lying on f1)
    '''
        
    free_symbols_f1 = f1.free_symbols
    free_symbols_f2 = f2.free_symbols

    variables = tuple(free_symbols_f1.intersection(free_symbols_f2))
    pvariables = tuple(sp.symbols(f'{symbol}_p') for symbol in variables)

    variables_dict = dict(zip(pvariables, variables))
    pvariables_dict = dict(zip(variables, pvariables))

    pgrad_f1 = gradient(f1.subs(pvariables_dict), pvariables)
    
    transform_1 = td_legendre_transform(variables, pgrad_f1, f2)
    transform_2 = td_legendre_transform(pvariables, pgrad_f1, f1.subs(pvariables_dict))

    discriminant = recursive_discriminant(transform_1 - transform_2, variables)
    return primitive(discriminant.subs(variables_dict), *variables)[1]


def bounding_points(points):
    '''
    Calculate bounding points in the projected space (excluding the last dimension).

    Parameters:
        points (array): Points in N-dimensional space, with the last dimension representing energy.

    Returns:
        bounding_points (array): Indices of points that define the convex hull in the projected space.
    '''
    if points.shape[1] == 1:
        bounding_points = np.array([np.argmin(points), np.argmax(points)])
    else:
        bounding_points = ConvexHull(points).simplices

    return bounding_points.flatten()

def lower_convex_hull(points):
    '''
    Calculate the lower convex hull, assuming the last dimension represents energy.

    Parameters:
        points (array): Points in N-dimensional space, with the last dimension representing energy.

    Returns:
        lower_hull (array): Array of indices describing the points that form the lower convex hull.
    '''
    processing_points = points.copy()

    projected_points = processing_points[:, :-1]
    bp = bounding_points(projected_points)
    
    fake_points = processing_points[bp].copy()
    fake_points[:, -1] += 10  # Small offset to create "upper" points
    processing_points = np.vstack((processing_points, fake_points))

    hull = ConvexHull(processing_points)
    simplices = hull.simplices

    mask = np.all(simplices < len(points), axis=1)
    lower_hull = simplices[mask]

    return lower_hull

