# Math and geometry libraries
import numpy as np
from scipy.spatial import ConvexHull
from sympy import Eq, solve, discriminant, diff, symbols


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
    fake_points[:, -1] += 500000  # Small offset to create "upper" points
    processing_points = np.vstack((processing_points, fake_points))

    hull = ConvexHull(processing_points)
    simplices = hull.simplices

    mask = np.all(simplices < len(points), axis=1)
    lower_hull = simplices[mask]

    return lower_hull


def projection_function(f1, f2):
    '''
    Computes the projection of f2 onto f1.
    
    Parameters:
        f1 : sympy expression

        f2 : sympy expression
    
    Returns:
        proj : sympy expression

        variables : tuple

        pvariables : tuple
    '''
    free_symbols_f1 = f1.free_symbols
    free_symbols_f2 = f2.free_symbols

    variables = tuple(free_symbols_f1.intersection(free_symbols_f2))
    pvariables = tuple(symbols(f'{symbol}_p') for symbol in variables)

    sum_terms = [(variable - pvariables[i])*diff(f1, variable).subs(variable, pvariables[i]) for i, variable in enumerate(variables)]
    proj = f2 - f1.subs(dict(zip(variables, pvariables))) - sum(sum_terms)

    return proj.expand(), variables, pvariables

def taylor_2nd_order(f, x, a):
    '''
    Computes the Taylor series approximation of f up to degree 2 centered at x = a.
    
    Parameters:
        f : sympy expression

        x : sympy symbol

        a : value
    
    Returns:
        sympy expression
    '''
    f_a = f.subs(x, a)
    f_prime = diff(f, x).subs(x, a) * (x - a)
    f_double_prime = diff(f, x, x).subs(x, a) * (x - a)**2 / 2
    
    return f_a + f_prime + f_double_prime

def boundary(f1, f2):
    '''
    computes the boundary equation for tangent hyperplane to f1 and f2
    
    Parameters:
        f1 : sympy expression

        f2 : sympy expression
    
    Returns:
        disc: sympy expression of the boundary condition
    '''
    proj, variables, pvariables = projection_function(f1, f2)

    sol_dict = dict()
    for i, variable in enumerate(variables):
        pvariable = pvariables[i]

        # This is the condition for the critical points
        condition = Eq(0, diff(f2, variable) - diff(f1, variable).subs(variable, pvariable))
        sols = solve(condition, variable)
        for sol in sols:
            M = taylor_2nd_order(proj, variable, sol.evalf())
            
            # Now we need to just compute the discriminant of the Function
            disc = discriminant(M.expand(), variable)

    return disc
