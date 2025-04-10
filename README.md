## Name
Hybrid storage plant optimization tool 

## Description
This is a tool for determining the optimal size and composition of the storage units of a hybrid energy storage plant, for a given set of requirements, constraints, environmental and market conditions.

## Visuals
(Module diagrams etc. go here)

## Installation
(Installation instructions to follow, including required modules etc.)

## Usage
(Usage instructions and example go here)

<!-- ## Authors and acknowledgment
Show your appreciation to those who have contributed to the project. -->

## License
tbd


## Explanation of sizing_Gradient.py 
Hybrid Energy Storage System Sizing Problem
Assumes 2 Energy Storage Systems: A and B
Optimizes Power Capacities: P_A, P_B and Energy Capacities E_A, E_B
Minimizes: Annualized Investment Cost (minus) Benefits
. Annualized Investment Cost = InvestmentCost / AnnuityFactor.
.. 1/AnnuityFactor = r(r+1)^L / ( (r+1)^L - 1)
.. InvestmentCost = c_A^P * P_A + c_B^P * P_B + c_A^E * E_A + c_B^E * E_B + c_F
..   where c_i^P, c_i^E cost for power/energy capacity and c_F a fixed cost
. Benefits are evaluated by running a Daily optimization problem for
.   a number of characteristic days (e.g., 12 days), with fixed capacities.

 The sizing problem can include bounds on budget or on capacities.

 The sizing problem is solved with a gradient method.
 - The derivative of the Investment Cost can be analytically obtained,
 - whereas the derivative of the Benefits will be evaluated (essntially it is a non-derivative method)
 - by perturbing each capacity and estimating the derivative using finite differences.

 Example of the Gradient Method:
 - The x array should include the capacities to be optimized,
 - i.e., x = [P_A, P_B, E_A, E_B]
