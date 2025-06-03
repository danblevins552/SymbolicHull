import sympy as sp
from sympy import symbols, diff, primitive, discriminant
import numpy as np
from scipy.spatial import ConvexHull
import json


# def quad_discriminant(expr, var):
#     """
#     Compute the discriminant of a quadratic expression with respect to a given variable.

#     Parameters:
#         expr (sympy expression): The quadratic expression.
#         var (sympy symbol): The variable to compute the discriminant with respect to.

#     Returns:
#         discriminant (sympy expression): The computed discriminant.
#     """
#     # Extract coefficients of x^2, x, and constant term
#     collection = sp.collect(expr, var)
#     a = collection.coeff(var, 2)  # Coefficient of x^2
#     b = collection.coeff(var, 1)  # Coefficient of x
#     c = collection.coeff(var, 0)  # Constant term

#     # Compute the discriminant manually
#     d = b**2 - 4*a*c

#     return d


def projection_function(f1, f2):
    """
    Compute the projection of one sympy expression onto another.

    Parameters:
        f1 (sympy expression): The function to project onto.
        f2 (sympy expression): The function to be projected.

    Returns:
        proj (sympy expression): The resulting projection expression.
        variables (tuple): The variables common to both functions.
        pvariables (tuple): The transformed projection variables.
    """
    free_symbols_f1 = f1.free_symbols
    free_symbols_f2 = f2.free_symbols

    variables = tuple(free_symbols_f1.intersection(free_symbols_f2))
    pvariables = tuple(symbols(f'{symbol}_p') for symbol in variables)

    sum_terms = [(variable - pvariables[i])*diff(f1, variable).subs(variable, pvariables[i]) for i, variable in enumerate(variables)]
    proj = f2 - f1.subs(dict(zip(variables, pvariables))) - sum(sum_terms)
    proj = proj.expand()

    return primitive(proj)[1], variables, pvariables


def recursive_discriminant(expr, vars):
    """
    Compute the discriminant recursively over multiple variables.
    The order of the variables shouldn't matter but can make sympy error out if the higher order discriminant is done first.

    Parameters:
        expr (sympy expression): The expression to compute the discriminant of.
        vars (list): List of variables to compute the discriminant with respect to.

    Returns:
        discriminant (sympy expression): The resulting discriminant expression.
    """

    #TODO: We need to sort the variables in order of lowes degree to highest degree when appearing in the expression

    vars_list = list(vars)
    def recursive_discriminant_helper(expr, vars):
        # if there are no variables, return the expression
        if vars == []:
            return expr
        
        for var in vars_list:
            if expr.has(var):
                vars_list.remove(var)
                return primitive(discriminant(recursive_discriminant(expr, vars_list), var))[1]

            
    return recursive_discriminant_helper(expr, vars_list)


def save_sympy_dict_to_json_repr(dictionary, filename):
    """
    Save a dictionary of sympy expressions to a JSON file using their `srepr()` representation.

    Parameters:
        dictionary (dict): Dictionary with sympy expressions as values.
        filename (str): The path to the file where data will be stored.

    Returns:
        None
    """   
    serializable_dict = {key: sp.srepr(value) for key, value in dictionary.items()}
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=4)


def load_sympy_dict_from_json(filename):
    """
    Load a JSON file and reconstruct sympy expressions from `srepr()` strings.

    Parameters:
        filename (str): The path to the JSON file.

    Returns:
        dict: Dictionary of reconstructed sympy expressions.
    """
    with open(filename, 'r') as f:
        raw_dict = json.load(f)
    
    # Use `sympy.sympify()` with a safe `locals` dictionary
    return {key: sp.sympify(raw_dict[key], locals=vars(sp)) for key in raw_dict}