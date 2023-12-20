# ANN-Based-Optimization-of-Airfoil-in-Transonic-Regime

In this present work, genetic algorithm is used to design the shape of an airfoil to minimize the drag in transonic regime. The flight condition of the airfoil is Ma=0.734. The design variables for the optimization process are PARSEC parameters and angle of attack. 
There are three constraints set on the optimization process. First constraint is the lift coefficient c_l should be at least 0.824. The second constraint is pitching moment coefficient c_m. The optimized airfoil shall not have high pitching moment coefficient. Hence, in this optimization process, the absolute value of the pitching moment coefficient should not exceed 0.093. The last constraint is the minimum airfoil thickness is 12% chord length. This constraint is set for structural and fuel tank placement consideration.

# How to Run
1. Install Pointwise Software.
2. Install required python libraries.
3. Defined the optimization parameters, i.e. upper bound, lower bound, objective function, and constraint functions.
4. Defined the flight condition.
