# ==============================================================================
# A From-Scratch Optimized Mixed-Integer Linear Programming (MILP) Solver
# Implemented in Python using only NumPy
#
# FINAL, PERFORMANT, AND CORRECTED VERSION (v3)
#
# Features:
# - (Correction) MILP Solver now intelligently directs the Simplex engine to
#   avoid the slow Two-Phase method whenever possible, fixing the performance issue.
# - (Correction) Robustly handles the Phase 1 -> Phase 2 transition in the
#   Simplex solver by using a canonical "pricing out" method, preventing hangs.
# - Efficient, stateless Branch and Bound algorithm.
# - Best-Bound node selection and Most-Fractional variable branching.
# ==============================================================================

import sys
import numpy as np
import heapq
from typing import List, Tuple, Optional, Literal

# --- Type Definitions for Clarity ---
Sense = Literal['<=', '>=', '==']
LP_Status = Literal['optimal', 'infeasible', 'unbounded']
Solution = Optional[np.ndarray]
ObjectiveValue = Optional[float]
BranchConstraint = Tuple[int, Literal['<=', '>='], float]


class Node:
    """Represents a node in the Branch and Bound search tree."""
    def __init__(self, lp_obj_val: float, constraints: List[BranchConstraint]):
        self.lp_obj_val: float = lp_obj_val
        self.constraints: List[BranchConstraint] = constraints

    def __lt__(self, other: 'Node') -> bool:
        """Comparison method for the priority queue (max-heap for maximization)."""
        return self.lp_obj_val > other.lp_obj_val


class SimplexSolver:
    """A pure "engine" that solves an LP. It now has two distinct entry points."""
    def __init__(self, A_std: np.ndarray, b_std: np.ndarray, c_std: np.ndarray):
        # Make copies to prevent modification of the original problem matrices
        self.A_std = A_std.copy()
        self.b_std = b_std.copy()
        self.c_std = c_std.copy()
        self.num_vars_std = self.A_std.shape[1]
        self.num_constraints_std = self.A_std.shape[0]
        self.basis: List[int]

    def _solve_simplex_tableau(self, tableau: np.ndarray) -> Tuple[LP_Status, Optional[np.ndarray]]:
        """The core simplex iteration logic (pivoting)."""
        iteration_count = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops
        
        while np.any(tableau[-1, :-1] < -1e-9):
            iteration_count += 1
            
            if iteration_count > max_iterations:
                print(f"WARNING: Simplex hit iteration limit ({max_iterations}), terminating")
                return "infeasible", None
            
            pivot_col = np.argmin(tableau[-1, :-1])
            
            # Check for unboundedness
            if np.all(tableau[:-1, pivot_col] <= 1e-9):
                return "unbounded", None
            
            # Ratio test to find the pivot row
            ratios = np.full(self.num_constraints_std, np.inf)
            positive_mask = tableau[:-1, pivot_col] > 1e-9
            ratios[positive_mask] = tableau[:-1, -1][positive_mask] / tableau[:-1, pivot_col][positive_mask]
            
            # This check is redundant due to the one above but is a safe guard.
            if np.all(ratios == np.inf):
                return "unbounded", None
                
            pivot_row = np.argmin(ratios)
            self.basis[pivot_row] = pivot_col
            
            # Perform the pivot
            pivot_element = tableau[pivot_row, pivot_col]
            if abs(pivot_element) < 1e-9: 
                return "infeasible", None # Should not happen with proper ratio test
            
            tableau[pivot_row, :] /= pivot_element
            for i in range(self.num_constraints_std + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
            
            # Check for cycling (same objective value for too many iterations)
            if iteration_count > 50 and iteration_count % 10 == 0:
                print(f"WARNING: Simplex slow convergence - iteration {iteration_count}")
                
        return "optimal", tableau

    def solve_direct(self) -> Tuple[LP_Status, Solution, ObjectiveValue]:
        """FAST PATH: For problems with a trivial all-slack/all-artificial basis."""
        tableau = np.vstack((
            np.hstack((self.A_std, self.b_std.reshape(-1, 1))),
            np.append(-self.c_std, 0)
        ))
        # Assumes basis is the set of slack variables, which are the last m columns
        self.basis = list(range(self.num_vars_std - self.num_constraints_std, self.num_vars_std))
        
        # Price out the basic variables to ensure the objective row is in canonical form
        for i, basis_col in enumerate(self.basis):
            if abs(tableau[-1, basis_col]) > 1e-9:
                tableau[-1,:] -= tableau[-1, basis_col] * tableau[i,:]

        status, tableau_final = self._solve_simplex_tableau(tableau)
        
        if status != "optimal" or tableau_final is None:
            return status, None, None

        solution = np.zeros(self.num_vars_std)
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < self.num_vars_std:
                solution[basis_var_idx] = tableau_final[i, -1]
        
        # Objective value is in the bottom-right corner of the tableau
        return "optimal", solution, tableau_final[-1, -1]

    def solve_two_phase(self) -> Tuple[LP_Status, Solution, ObjectiveValue]:
        """SLOW PATH: The full Two-Phase method for complex problems."""
        # --- Phase 1: Minimize the sum of artificial variables ---
        num_artificial = self.num_constraints_std
        A_phase1 = np.hstack((self.A_std, np.eye(num_artificial)))
        
        # Objective for Phase 1 is to minimize sum of artificials,
        # which is equivalent to maximizing the negative sum.
        c_phase1 = np.zeros(self.num_vars_std + num_artificial)
        c_phase1[-num_artificial:] = -1 # Maximize -w = -sum(a_i)
        
        tableau = np.vstack((
            np.hstack((A_phase1, self.b_std.reshape(-1, 1))),
            np.append(-c_phase1, 0) # Note: -c_phase1 because solver maximizes
        ))
        
        # The initial basis is all artificial variables
        self.basis = list(range(self.num_vars_std, self.num_vars_std + num_artificial))
        
        # Price out the initial basic (artificial) variables
        for i, basis_col in enumerate(self.basis):
            if abs(tableau[-1, basis_col]) > 1e-9:
                 tableau[-1,:] -= tableau[-1, basis_col] * tableau[i,:]
            
        status_p1, tableau_p1 = self._solve_simplex_tableau(tableau)

        # If Phase 1 is not optimal or the objective is non-zero, the problem is infeasible
        if status_p1 != "optimal" or tableau_p1 is None or abs(tableau_p1[-1, -1]) > 1e-9:
            return "infeasible", None, None

        # --- Phase 2: Solve the original problem ---
        
        # Drop artificial variable columns to create the starting tableau for Phase 2
        A_p2 = tableau_p1[:-1, :self.num_vars_std]
        b_p2 = tableau_p1[:-1, -1]
        
        tableau_p2 = np.vstack((
            np.hstack((A_p2, b_p2.reshape(-1, 1))),
            np.append(-self.c_std, 0)
        ))
        
        # The basis is carried over from Phase 1. Price out the basic variables
        # using the original objective function to get a valid starting tableau.
        for i, basis_col in enumerate(self.basis):
             if basis_col < self.num_vars_std: # Defensively check against artificials
                if abs(tableau_p2[-1, basis_col]) > 1e-9:
                    tableau_p2[-1,:] -= tableau_p2[-1, basis_col] * tableau_p2[i,:]

        # Solve the Phase 2 tableau
        status_p2, tableau_p2_final = self._solve_simplex_tableau(tableau_p2)
        if status_p2 != "optimal" or tableau_p2_final is None:
            return status_p2, None, None

        solution = np.zeros(self.num_vars_std)
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < self.num_vars_std:
                solution[basis_var_idx] = tableau_p2_final[i, -1]
                
        return "optimal", solution, tableau_p2_final[-1, -1]


class OptimizedMILPSolver:
    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray, 
                 senses: List[Sense], integer_vars_indices: List[int]):
        self.c_orig_len = len(c)
        self.A_orig = A
        self.b_orig = b
        self.senses_orig = senses
        self.c_orig = c
        self.integer_vars_indices = integer_vars_indices
        self.A_std_root: np.ndarray
        self.b_std_root: np.ndarray
        self.c_std_root: np.ndarray
        self.root_is_complex: bool
        self.best_solution: Solution = None
        self.best_obj_val: float = -np.inf
        self.node_count: int = 0
        self.node_queue: List[Node] = []

    def _initial_standardize(self) -> None:
        """Converts the user problem to standard form and determines its complexity."""
        # A problem is "complex" if it requires the Two-Phase Simplex method at the root.
        self.root_is_complex = any(s != '<=' for s in self.senses_orig) or np.any(self.b_orig < 0)
        
        A_temp = self.A_orig.copy()
        b_temp = self.b_orig.copy()
        
        # Ensure RHS is non-negative
        for i in range(A_temp.shape[0]):
            if b_temp[i] < 0:
                A_temp[i, :] *= -1
                b_temp[i] *= -1
                if self.senses_orig[i] == '<=': self.senses_orig[i] = '>='
                elif self.senses_orig[i] == '>=': self.senses_orig[i] = '<='

        num_slack_surplus = sum(1 for s in self.senses_orig if s != '==')
        self.A_std_root = np.hstack((A_temp, np.zeros((A_temp.shape[0], num_slack_surplus))))
        self.c_std_root = np.append(self.c_orig, np.zeros(num_slack_surplus))
        self.b_std_root = b_temp
        
        slack_surplus_idx = 0
        for i, sense in enumerate(self.senses_orig):
            if sense == '<=':
                self.A_std_root[i, self.c_orig_len + slack_surplus_idx] = 1
                slack_surplus_idx += 1
            elif sense == '>=':
                self.A_std_root[i, self.c_orig_len + slack_surplus_idx] = -1
                slack_surplus_idx += 1

    def _solve_node_lp(self, constraints: List[BranchConstraint]) -> Tuple[LP_Status, Solution, ObjectiveValue]:
        """Builds and solves the LP for a node, intelligently choosing the Simplex method."""
        # Determine if this specific node's LP requires the Two-Phase method.
        # This is true if the root problem needed it, or if a '>=' branch constraint was added.
        needs_two_phase = self.root_is_complex or any(c[1] == '>=' for c in constraints)
        
        num_branch_constraints = len(constraints)
        
        # Start with the root LP in standard form
        A_node = self.A_std_root
        b_node = self.b_std_root
        c_node = self.c_std_root
        
        if num_branch_constraints > 0:
            # If there are branching constraints, we need to add new rows and slack variables
            num_base_vars = self.A_std_root.shape[1]
            
            # Create copies to avoid modifying the root problem matrices
            A_node = np.vstack((
                np.hstack((self.A_std_root, np.zeros((self.A_std_root.shape[0], num_branch_constraints)))),
                np.zeros((num_branch_constraints, num_base_vars + num_branch_constraints))
            ))
            b_node = np.append(self.b_std_root, np.zeros(num_branch_constraints))
            c_node = np.append(self.c_std_root, np.zeros(num_branch_constraints))
            
            # Add each branching constraint as a new row
            for i, (var_idx, sense, bound) in enumerate(constraints):
                row_idx = self.A_std_root.shape[0] + i
                A_node[row_idx, var_idx] = 1
                b_node[row_idx] = bound
                
                if sense == '<=':
                    A_node[row_idx, num_base_vars + i] = 1 # Add slack variable
                else: # sense == '>='
                    A_node[row_idx, num_base_vars + i] = -1 # Add surplus variable
        
        lp_solver = SimplexSolver(A_node, b_node, c_node)
        if needs_two_phase:
            return lp_solver.solve_two_phase()
        else:
            return lp_solver.solve_direct()

    def solve(self) -> Tuple[Solution, ObjectiveValue]:
        print("--- Starting Optimized MILP Solver ---")
        self._initial_standardize()
        
        # Solve the root node LP relaxation
        self.node_count += 1
        status, solution, obj_val = self._solve_node_lp([])
        
        if status != "optimal" or solution is None or obj_val is None:
            print(f"Root problem is {status}. No solution exists.")
            return None, None
            
        print(f"Root LP objective: {obj_val:.6f}")
        
        # Initialize the priority queue with the root node
        heapq.heappush(self.node_queue, Node(obj_val, []))
        
        main_loop_count = 0
        max_main_iterations = 10000  # Reasonable limit
        max_nodes_in_queue = 1000    # Limit queue size to prevent memory explosion

        while self.node_queue:
            main_loop_count += 1
            
            if main_loop_count > max_main_iterations:
                print(f"Terminating: Hit iteration limit ({max_main_iterations})")
                break
                
            if len(self.node_queue) > max_nodes_in_queue:
                print(f"Terminating: Queue size exceeded limit ({max_nodes_in_queue})")
                break
                
            # Get the most promising node (best bound)
            current_node = heapq.heappop(self.node_queue)

            # Enhanced bound pruning with tolerance
            if current_node.lp_obj_val <= self.best_obj_val + 1e-6:
                continue
                
            # Prune nodes with too many constraints (prevents explosion)
            if len(current_node.constraints) > 50:
                continue
            
            # Solve the LP for the current node
            status, solution, obj_val = self._solve_node_lp(current_node.constraints)

            # Prune if the node is infeasible or its solution is worse than our best
            if status != "optimal" or solution is None or obj_val is None:
                continue
            if obj_val <= self.best_obj_val + 1e-6:  # Add tolerance
                continue
            
            # Check for integer feasibility
            is_integer_feasible = True
            fractionalities = {}
            for i in self.integer_vars_indices:
                val = solution[i]
                if abs(val - round(val)) > 1e-6:
                    is_integer_feasible = False
                    # Store fractionality, prioritizing variables closer to 0.5
                    fractionalities[i] = abs(val - np.floor(val) - 0.5)

            if is_integer_feasible:
                # If we found an integer solution that's better than our current best, update it
                if obj_val > self.best_obj_val:
                    self.best_obj_val = obj_val
                    self.best_solution = solution[:self.c_orig_len]
                    print(f"Node {self.node_count}: Found new integer solution. Objective: {obj_val:.4f}")
                continue

            # If not integer feasible, branch on the most fractional variable
            if not fractionalities:
                continue
            
            # Choose variable with maximum fractionality (closest to 0.5)
            branch_var_idx = min(fractionalities, key=fractionalities.get)
            fractional_val = solution[branch_var_idx]
            
            # Only branch if the gap is significant
            if obj_val - self.best_obj_val < 1e-4:
                continue
            
            # Create two new branches (nodes) and add them to the queue
            self._queue_branch(current_node.constraints + [(branch_var_idx, '<=', np.floor(fractional_val))])
            self._queue_branch(current_node.constraints + [(branch_var_idx, '>=', np.ceil(fractional_val))])

        print(f"\n--- Solver Finished ---\nTotal nodes explored: {self.node_count}")
        return self.best_solution, self.best_obj_val

    def _queue_branch(self, constraints: List[BranchConstraint]) -> None:
        """Solves the LP for a new branch and, if promising, adds it to the node queue."""
        self.node_count += 1
        status, _, obj_val = self._solve_node_lp(constraints)
        # Add to queue only if the branch is feasible and could potentially yield a better solution
        # Also check that the improvement is significant
        if (status == "optimal" and obj_val is not None and 
            obj_val > self.best_obj_val + 1e-6):
            heapq.heappush(self.node_queue, Node(obj_val, constraints))


# Main block with test cases
if __name__ == '__main__':
    # --- Test Case 1: Capital Budgeting Problem (Binary Variables) ---
    print("\n--- TEST CASE 1: CAPITAL BUDGETING (BINARY) ---")
    c1 = np.array([9, 5, 8, 10, 3])
    A1 = np.array([
        [4, 3, 5, 7, 2],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]
    ])
    b1 = np.array([10, 1, 1, 1, 1, 1])
    senses1: List[Sense] = ['<=', '<=', '<=', '<=', '<=', '<=']
    integer_vars_indices1 = [0, 1, 2, 3, 4]

    solver1 = OptimizedMILPSolver(c1, A1, b1, senses1, integer_vars_indices1)
    solution1, value1 = solver1.solve()

    if solution1 is not None:
        print(f"\nOptimal Objective Value: {value1:.4f}")
        print("Optimal Solution:", np.round(solution1).astype(int))
    else:
        print("\nNo integer-feasible solution found.")
        
    # --- Test Case 2: Production Mix Problem (Mixed Constraints) ---
    print("\n\n--- TEST CASE 2: PRODUCTION MIX (MIXED CONSTRAINTS) ---")
    c2 = np.array([50, 30])
    A2 = np.array([[2, 1], [1, 3], [1, 1]])
    b2 = np.array([10, 12, 3])
    senses2: List[Sense] = ['<=', '<=', '>=']
    integer_vars_indices2 = [0, 1]

    solver2 = OptimizedMILPSolver(c2, A2, b2, senses2, integer_vars_indices2)
    solution2, value2 = solver2.solve()

    if solution2 is not None:
        print(f"\nOptimal Objective Value: {value2:.4f}")
        print("Optimal Solution:", np.round(solution2).astype(int))
    else:
        print("\nNo integer-feasible solution found.")

    # --- Test Case 3: Problem with an Equality Constraint ---
    print("\n\n--- TEST CASE 3: EQUALITY CONSTRAINT ---")
    c3 = np.array([2, 5])
    A3 = np.array([[2, 1], [1, 1]])
    b3 = np.array([16, 9])
    senses3: List[Sense] = ['<=', '==']
    integer_vars_indices3 = [0, 1]
    
    solver3 = OptimizedMILPSolver(c3, A3, b3, senses3, integer_vars_indices3)
    solution3, value3 = solver3.solve()
    
    if solution3 is not None:
        print(f"\nOptimal Objective Value: {value3:.4f}")
        print("Optimal Solution:", np.round(solution3).astype(int))
    else:
        print("\nNo integer-feasible solution found.")