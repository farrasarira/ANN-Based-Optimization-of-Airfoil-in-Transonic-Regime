# ANN-Based-Optimization-of-Airfoil-in-Transonic-Regime

The performance of an airplane is directly related to its aerodynamic forces, including lift force and drag force which will be influenced by determining the airfoil that will be used on the wing, so there is a need to optimize the airfoil, especially the shape of the airfoil to optimize the lift and drag power. Initially, airfoil optimization was carried out using a trial-and-error method using an expensive and inefficient wind tunnel, then it was developed using Computational Fluid Dynamics (CFD) simulation. Recently, advanced computer technology can optimize the shape of airfoils using machine learning. 
Airfoil with various shapes is available in the literature. To optimize the existing airfoil, a technique for modifying the airfoil shape is needed through the parameterization method. Currently there are many parameterization methods, but the most common and widely used in many studies is the PARSEC method. The PARSEC method uses 11 parameters to represent airfoil geometry (thickness, curvature, maximum thickness abscissa, etc.).
Machine Learning is a part of artificial intelligence that gives computers the ability to learn from data to be able to predict optimal solutions with simple programming. Optimization using machine learning is very dependent on the data and methods used, to determine the right method it is necessary to carry out simulations using several type methods. However, this research will only use Artificial Neural Networks by varying the activation function, number of neural layers and number of hidden layers to determine the best method. Furthermore, the data used uses CFD simulation data by determining the number of data sets for the simulation using Latin hypercube sampling (LHS). In creating simulation data using 11 parameters of the PARSEC Method to represent airfoil geometry (thickness, curvature, maximum thickness abscissa, etc.) and angle of attack as input, Lift Coefficient (Cl) and Moment Coefficient (Cm) as Limits with the aim of becoming the Drag Coefficient (CD ).

# How to Run
1. Install Pointwise Software.
2. Install required python libraries.
3. Defined the optimization parameters, i.e. upper bound, lower bound, objective function, and constraint functions.
4. Defined the flight condition.


