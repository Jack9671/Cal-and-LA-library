
import random
import math
import copy
import numpy as np
import sympy as sp 
import mpmath as mp
import time

def main():
    print("Welcome to the Linear Algebra Library")
    print("feel free to play with the class") 
    #example of how to use the class  
    A = np.array([[0, 1],
                 [-3, 4]])
    diagonalize(A, verbose = True)
 

def jordan_normal_form(A: np.ndarray, verbose: bool = False) -> list[np.ndarray] | None:
    #initialize the matrix P
    P = np.zeros((len(A),0)) #0 because we will append the columns of P later
    #initialize the matrix J
    J = np.zeros((len(A),len(A)))
    #used when A has only one eigenvalue
    evalues , evectors = np.linalg.eig(A)
    evalues = np.round(np.real(evalues),10)
    evectors = np.round(np.real(evectors),10)
    #debugging part
    if verbose:
        print("evalues is: ",evalues)
        print("evectors is: ")
        print_matrix(evectors)
    #end
    
    distict_evalues_and_their_algebraic_multiplicity = dict()
    for eigenval in evalues:
        if eigenval in distict_evalues_and_their_algebraic_multiplicity:
            distict_evalues_and_their_algebraic_multiplicity[eigenval] += 1
        else:
            distict_evalues_and_their_algebraic_multiplicity[eigenval] = 1 
    #Step 1: iterate through each eigenvalue and its algebraic multiplicity to find the jordan chains whose sum of col vectors of each chain = algebraic multiplicity
    for eigenval, algebraic_multiplicity in distict_evalues_and_their_algebraic_multiplicity.items():
        if algebraic_multiplicity == 1:
            P = np.column_stack((P,evectors[:,np.where(evalues == eigenval)[0][0]]))
        elif algebraic_multiplicity > 1:
            if verbose:
                print(f"for eigenval = {eigenval} and algebraic_multiplicity = {algebraic_multiplicity}")
            jordan_chain = _compute_jordan_chains(A,eigenval,algebraic_multiplicity, verbose = verbose)
            for chain in jordan_chain:
                P = np.column_stack((P,chain))

    #Step2: Find the inverse of P
    P_inv = np.linalg.inv(P)
    #Step3: Find the J matrix
    J = np.dot(np.dot(P_inv,A),P)
    if verbose:
        print("The matrix P is: ")
        print_matrix(P)
        print("The matrix J is: ")
        print_matrix(J)
        print("The matrix P^-1 is: ")
        print_matrix(P_inv)
        print("Check whether the Jordan Normal Form is correct or not: ")
        print("PJP^-1 is: ")
        print_matrix(np.dot(np.dot(P, J), P_inv))
    return [P,J,P_inv]
    
def diagonalize(A: np.ndarray, verbose: bool = False) -> list[np.ndarray] | None:
    #main idea: A = PDP^-1 where D is a diagonal matrix containing linearly independent evalues and P is a matrix whose columns are envectors of A
    try: # works if D contains linearly independent evalues
        evalues, evectors_matrix = np.linalg.eig(A)
        D = np.diag(evalues)
        #round evector_matrix to 0 if the value > 1e-10
        evectors_matrix = np.round(evectors_matrix,10)
        evectors_matrix_inv = np.linalg.inv(evectors_matrix)
        if verbose:
            print("P: ")
            print_matrix(evectors_matrix)
            print("D: ")
            print_matrix(D)
            print("P^-1: ")
            print_matrix(evectors_matrix_inv)
            print("Check whether the diagonalization is correct or not: ")
            print("PDP^-1 is: ")
            print_matrix(np.dot(np.dot(evectors_matrix, D), evectors_matrix_inv))
        return [evectors_matrix, D, evectors_matrix_inv]
    except np.linalg.LinAlgError: 
        print("The matrix is not diagonalizable\n we find the Jordan form instead")
        if verbose:
                evalues , evectors = np.linalg.eig(A)
                #evalues = np.round(np.real(evalues),3)
                #evectors = np.round(np.real(evectors),3)
                #debugging part
                print("evalues is: ",evalues)
                print("evectors is: ")
                print_matrix(evectors)
                #end
        return jordan_normal_form(A, verbose = verbose)

def det(A: np.ndarray, verbose = False) -> float:
    U = pldu_factor(A)[2]
    permutation_parity = pldu_factor(A)[3]
    determinant = 1 #default
    for i in range(len(U)):
        determinant *= U[i][i]
    determinant *= permutation_parity
    if verbose:
        print(f"The determinant of the matrix is: {determinant}")
    return determinant

def rref(matrix: np.ndarray, verbose: bool = False) -> np.ndarray| None: #Reduced Row Echelon Form
    #information of the matrix [A]
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm: Gauss-Jordan elimination
    #STEP1: Gaussian Elimination
    matrix = gauss_eliminate(matrix, verbose=verbose)
    #STEP2: Jordan Elimination
    matrix = jordan_eliminate(matrix, verbose = verbose)
    #STEP3: Normalize the pivots to 1 to get R
    for y_index in range(num_of_Y): 
        for x_index in range(y_index, num_of_X): #check from left to right to find the non-0 pivot of upper or semi-upper triangular matrix
            if matrix[y_index][x_index] != 0:
                pivot_normalized_factor = matrix[y_index][x_index]
                matrix[y_index] = matrix[y_index] / pivot_normalized_factor #normalize the pivot to 1
                break #After normalizing the current y_index, break the inner loop to move to the next y_index    
    if verbose:
        print("After normalization, R is:")
        print_matrix(matrix)
    #STEP6: Return the R matrix
    return matrix

def pldu_factor(matrix: np.ndarray, get_D_or_not: bool = False, verbose: bool = False) -> list[np.ndarray] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #additional part for finding the determinant
    permutation_parity = 0
    #main algorithm
    #Step1: Initialize the L matrix and a column matrix P containing the index of the row
    L_matrix = np.identity(num_of_Y)
    P_row_index_storage_matrix = [y for y in range(num_of_Y)]
    Arr_of_steps_of_permutation = [0 for y in range(num_of_Y)]
    #Step2: Modified Version of Gaussian Elimination for purpose of finding P and L
    ## The idea is to transform the matrix into an upper triangular matrix
    for x_index in range(num_of_X-1):  # note: customizable or we do not need to find the pivot of the last column of a square matrix
        #main algorithm of getting elimination matrix
        #STEP1: initialize the elimination matrix
        eli_matrix = np.identity(num_of_Y)
        #STEP2: check whether current pivot matrix[x_index][x_index] is 0 or not, if it is 0, exchange and ready for elimination
        y_index = x_index
        if matrix[y_index][x_index] == 0: #if the current y is 0, then exchange the y with the next y that has non-0 pivot
           y_index_of_non0_pivot = _find_non0_pivot(matrix, x_index, y_index)
           permutation_parity += 1
           if y_index_of_non0_pivot != None:
               matrix[[y_index,y_index_of_non0_pivot]] = matrix[[y_index_of_non0_pivot,y_index]] # swap the rows
               L_matrix[y_index], L_matrix[y_index_of_non0_pivot] = L_matrix[y_index_of_non0_pivot], L_matrix[y_index] # swap the rows
               Arr_of_steps_of_permutation.append(y_index)
               Arr_of_steps_of_permutation.append(y_index_of_non0_pivot)
           elif y_index_of_non0_pivot == None:
               print("There exists a zero row during Gaussian Elimination") #if there is no non-0 pivot, then the matrix is singular
               return None
        #STEP3: elimination process for each y down the current y
        for y_index in range(x_index + 1,num_of_Y):#plus 1 because we do not need to eliminate the current row where the current pivot is located
                multipler = matrix[y_index][x_index]/(matrix[x_index][x_index])#
                L_matrix[y_index][x_index] = multipler
                eli_matrix[y_index][x_index] = -(multipler) 
        #then we get eli_matrix
        matrix = np.dot(eli_matrix, matrix)
    #check if the last y_index contains all 0 after elimination, if yes, then the system is singular
    if all(x == 0 for x in matrix[num_of_Y-1]):
        print("There exists a last all-0 row during Gaussian Elimination") if verbose else None
        return None
    #Step3: Set P_row_index_storage_matrix from Arr_of_steps_of_permutation
    for y1 in (len(Arr_of_steps_of_permutation)-1,0,-2):
        y2 = y1 -1
        P_row_index_storage_matrix[Arr_of_steps_of_permutation[y1]], P_row_index_storage_matrix[Arr_of_steps_of_permutation[y2]] = P_row_index_storage_matrix[Arr_of_steps_of_permutation[y2]], P_row_index_storage_matrix[Arr_of_steps_of_permutation[y1]]
    #Step4: get overall P matrix from P_row_index_storage_matrix 
    P_matrix = np.zeros((num_of_Y,num_of_Y))
    for y_index in range(len(P_row_index_storage_matrix)):
        P_matrix[y_index][P_row_index_storage_matrix[y_index]] = 1
    #step6: (optional) get D matrix by normalizing the pivots of U to 1
    if get_D_or_not:
        D_matrix = np.zeros((num_of_Y,num_of_X))
        for y_index in range(num_of_Y):
            pivot_normalized_factor = matrix[y_index][y_index]
            D_matrix[y_index][y_index] = pivot_normalized_factor
            matrix[y_index] = matrix[y_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
    #Step7: check permutation_parity
    if permutation_parity % 2 == 0:
        permutation_parity = 1
    elif permutation_parity % 2 == 1:
        permutation_parity = -1
    if verbose:
        print("The P matrix is:")
        print_matrix(P_matrix)  
        print("The L matrix is:")
        print_matrix(L_matrix)
        if get_D_or_not:
            print("The D matrix is:")
            print_matrix(D_matrix)
        print("The U matrix is:")
        print_matrix(matrix)
        if get_D_or_not:
            print ("Check whether the PLDU factorization is correct or not:")
            print("P*L*D*U is:")
            print_matrix(np.dot(np.dot(np.dot(P_matrix, L_matrix), D_matrix), matrix))
        else:
            print ("Check whether the PLU factorization is correct or not:")
            print("P*L*U is:")
            print_matrix(np.dot(np.dot(P_matrix, L_matrix), matrix))
        print(f"The permutation parity is: {permutation_parity}")
    if get_D_or_not:
        return [P_matrix, L_matrix, D_matrix, matrix, permutation_parity]
    else:
        return [P_matrix, L_matrix, matrix, permutation_parity]

def get_projection_vector(A: np.ndarray, vector_b: np.ndarray, verbose = False) -> np.ndarray: #project b onto A
    #P = A.(A^T.A)-1.A^T
    # A: mxn where m > n
    middle = np.linalg.inv(np.dot(np.transpose(A),A))
    P = np.dot(A,np.dot(middle,np.transpose(A)))
    vector_p = np.dot(P,vector_b)
    if verbose:
        print("The projection vector is:")
        print_matrix(vector_p)
        print("The error vector is:")
        error = vector_b - vector_p #error = b - p
        print_matrix(error)
    return vector_p
def get_projection_matrix(A): # A contains independent columns that span the subspace where a vector is projected onto
   # P = A.(A^T.A)-1.A^T
   ##get (A^T.A)-1
   middle = np.linalg.inv(np.dot(np.transpose(A),A))
   P = np.dot(np.dot(A,middle),np.transpose(A))
   return P     

def get_least_square_solution(A: np.ndarray, vector_b: np.ndarray, verbose = False) -> np.ndarray: #get x* #A: mxn where m > n
    '''
    There are 2 cases, A^T.A is invertible and not invertible
    '''
    ##CASE 1: A^T.A is invertible
    #x = (A^T.A)-1 . A^T.b in C(A^T)
    try:
        left_part = np.linalg.inv(np.dot(np.transpose(A),A))
        right_part = np.dot(np.transpose(A),vector_b)
        x = np.dot(left_part,right_part)
        if verbose:
            print("Case1: The least square solution is:")
            for i in range(len(x)):
                print(f"X{i+1} = {round(x[i][0],3)}")
        return x
    except np.linalg.LinAlgError:
        ##CASE 2: A^T.A is not invertible
        #Find x in A^T.A.x = A^T.b in C(A^T)
        #Step1: get A^T.A
        AtA = np.dot(np.transpose(A),A)
        #Step2: get A^T.b
        Atb = np.dot(np.transpose(A),vector_b)
        #Step3: Append A^tb to A^tA to form an augmented matrix [A^tA A^tb] then RREF it
        RREF_matrix = rref(np.column_stack((AtA, Atb)))
        #Step4: initialize the solution vector x by 0 
        x = np.zeros((len(AtA[0]),1))
        #Step6: Scan for pivot columns.
        '''
        The idea for step 6 is to set all entries of 
        col-vector-solution x hat 
        whose y_index matches the pivot row (y_index*) of RREF_matrix 
        to be equal to RREF_matrix[y_index*][last_x_index_of_RREF_matrix] or the "A^tb" part of RREF([A^tA A^tb]) 
        '''
        num_of_Y = len(AtA)
        num_of_X = len(AtA[0]) 
        last_x_index_of_RREF_matrix = len(RREF_matrix[0]) - 1
        all_0_row = False
        for y_index in range(num_of_Y):
            #check whether the current row is all 0 before finding the pivot column  
            all_0_row = all(x == 0 for x in RREF_matrix[y_index]) # not necessary to check the last x ( containing vector A^tb) like RREF_matrix[y_index][:last_x_index_of_RREF_matrix]). The reason is A^tb lies in the C(A^T) so A^tb will have same row 0 as A^T.A when RREF takes place
            if all_0_row == True:
                break 
            #if not, then proceed
            for x_index in range(num_of_X):
                if RREF_matrix[y_index][x_index] != 0:
                    x[y_index][0]= RREF_matrix[y_index][last_x_index_of_RREF_matrix]
                    break #break the inner loop if successfully find one solution entry 
        if verbose:
            print("Case2: The least square solution is:")
            for i in range(len(x)):
                print(f"X{i+1} = {round(x[i][0],3)}")    
        return x

def gauss_eliminate(matrix: np.ndarray, max_x_index_to_stop_eliminate: int = None, stop_if_found_no_pivot_for_a_column: bool = False, verbose: bool = False) -> np.ndarray | None:
    #information of the matrix
    matrix = copy.deepcopy(matrix)
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    last_x_index = num_of_X - 1
    if max_x_index_to_stop_eliminate == None or max_x_index_to_stop_eliminate > last_x_index: #None means users want to eliminate all columns, and the case > last_x_index is to prevent out of range error
       max_x_index_to_stop_eliminate = num_of_X-1 
    ## Clear the entries below the pivot
    for y_index in range(num_of_Y-1): # no need to eliminate the last row
        for x_index in range(y_index, max_x_index_to_stop_eliminate + 1):  
            eli_matrix = _get_elimination_matrix_for_a_column(matrix, x_index, y_index, "down", verbose) # "down" means to clear y_entries on particualr x_index down the current y_index
            if eli_matrix is None: # if none, skip to next x
                if stop_if_found_no_pivot_for_a_column: # recommended for invertibility checking and solving a system of equations
                    print("The matrix is singular") if verbose else None
                    return None
                continue # end current iteration and skip to the next iteration, which is next x
            #check if eli_matrix == identity matrix, if yes, then skip to the next x
            elif np.array_equal(eli_matrix, np.identity(num_of_Y)):
                break # if it is true, it means there is all 0 below the pivot, then increment y to the next y
            elif eli_matrix is not None:
                #after this, we get eli_matrix
                matrix = np.dot(eli_matrix, matrix)
                #ADDITIONAL PART FOR BETTER NUMERICAL STABILITY: ROUND EXTREME SMALL VALUES e^-10 TO 0
                #matrix = round_small_values_to_zero(matrix)
                if verbose: #print the step-by-step process for educational purposes if allowed
                    print(f"the elimination matrix for the {x_index+1}th collumn:")
                    print_matrix(eli_matrix)
                    print(f"the result:")
                    print_matrix(matrix)
                break # if pivot is found in the loop, then stop and increment y to the next y
    return matrix 

def jordan_eliminate(matrix: np.ndarray, max_x_index_to_stop_eliminate: int = None ,verbose: bool = False) -> np.ndarray: #Warning: only used after Gaussian Elimination is performed
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    max_rank = min(num_of_Y, num_of_X)
    if max_x_index_to_stop_eliminate == None or max_x_index_to_stop_eliminate > num_of_X: #None means users want to eliminate all columns, and the case > last_x_index is to prevent out of range error
       max_x_index_to_stop_eliminate = num_of_X-1
    #increment by 1 from second column to the last column specified by max_x_index_to_stop_eliminate
    for y_index in range (1, num_of_Y): 
        for x_index in range (max_x_index_to_stop_eliminate + 1):
            #check from left to right to find the first non-0 pivot
            # if found, then clear the entries above the pivot
            # if not found on the whole row or y_index, then increment to the next y_index
            if matrix[y_index][x_index] != 0:
                eli_matrix = _get_elimination_matrix_for_a_column(matrix, x_index, y_index, "up") # "up" means to clear y_entries on particualr x_index up the current y_index
                #check if eli_matrix == identity matrix, if yes, then skip to the next y
                if np.array_equal(eli_matrix, np.identity(num_of_Y)):
                    break
                matrix = np.dot(eli_matrix, matrix)
                if verbose: 
                     print(f"Jordan Step: the elimination matrix for the {x_index+1}th collumn: ")
                     print_matrix(eli_matrix)
                     print(f"Jordan Step: the result: ")
                     print_matrix(matrix)
                break #break the inner loop if the pivot is found
        if (y_index+1) == max_rank: #if the max rank is reached, then break the outer loop for saving computation
            break
    return matrix

def compute_nullspace(matrix):
    """
    Compute the null space of a given NumPy array and return it as a NumPy array.

    Parameters:
    matrix (np.ndarray): Input NumPy array.

    Returns:
    np.ndarray: Null space of the input matrix as a NumPy array, with basis vectors as columns.
    """
    # Convert the NumPy array to a SymPy Matrix
    sympy_matrix = sp.Matrix(matrix)
    
    # Compute the null space
    nullspace = sympy_matrix.nullspace()
    
    # Convert the null space vectors to a NumPy array and stack them as columns
    nullspace_array = np.hstack([np.array(vec).astype(float) for vec in nullspace])
    
    return nullspace_array


def _find_non0_pivot(matrix: np.ndarray, x_index: int, y_starting_index: int): 
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    for y_index in range(y_starting_index + 1,num_of_Y): #plus 1 because we are looking for the next lower Y after finding the 0 pivot of the current Y
        if matrix[y_index][x_index] != 0:
            return y_index
        y_index+=1
    #otherwise, return None if there is no non-0 pivot
    return None

def _get_elimination_matrix_for_a_column(matrix: np.ndarray, x_index: int, y_index: int, Up_or_Down_the_current_y_index: str, verbose: bool = False) -> np.ndarray | None: #eliminate the y_index on the particular x_index
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    #STEP1: initialize the elimination matrix
    eli_matrix = np.identity(num_of_Y)
    #STEP2: find the largest pivot and detect whether the column is all 0 or not
    y_index_of_largest_pivot = _find_largest_pivot_as_well_as_non_zero_pivot(matrix, x_index, y_index)
    if y_index_of_largest_pivot == y_index:
        pass
    elif y_index_of_largest_pivot != None:
        matrix[[y_index,y_index_of_largest_pivot]] = matrix[[y_index_of_largest_pivot,y_index]] # swap the rows
        if verbose == True:
               print(f"Partial Pivoting: Exchange row {y_index+1} with row {y_index_of_largest_pivot+1}:")
               print_matrix(matrix)
    elif y_index_of_largest_pivot == None: 
        return None
    #STEP3: elimination process for each y down the current y or up the current y
    if Up_or_Down_the_current_y_index.lower() == "down":
            multiplier = 1 #default
            y_index_of_current_pivot = y_index
            current_y_index = y_index + 1#plus 1 because we do not need to eliminate the current row where the current pivot is located
            while current_y_index < num_of_Y:
                multipler = -(matrix[current_y_index][x_index])/(matrix[y_index_of_current_pivot][x_index])#
                eli_matrix[current_y_index][y_index_of_current_pivot] = multipler 
                current_y_index+=1
    elif Up_or_Down_the_current_y_index.lower() == "up":
            multiplier = 1 #default
            y_index_of_current_pivot = y_index
            current_y_index = y_index -1 #- 1 because we do not need to eliminate the current row where the current pivot is located
            while current_y_index >= 0:
                multipler = -(matrix[current_y_index][x_index])/(matrix[y_index_of_current_pivot][x_index])#
                eli_matrix[current_y_index][y_index_of_current_pivot] = multipler 
                current_y_index-=1
    return eli_matrix
def _find_largest_pivot_as_well_as_non_zero_pivot(matrix: np.ndarray, x_index: int, y_starting_index: int):
    # Information of the matrix
    num_of_Y = len(matrix)
    
    # Main algorithm
    largest_pivot = matrix[y_starting_index][x_index]
    y_index_of_largest_pivot = y_starting_index
    
    for y_index in range(y_starting_index+1, num_of_Y):
        pivot_value = abs(matrix[y_index][x_index]) 
        # Check if this is the largest pivot found so far
        if pivot_value > largest_pivot:
            largest_pivot = pivot_value
            y_index_of_largest_pivot = y_index    
    # If no non-zero pivot was found, return None
    if largest_pivot == 0:
        return None
    
    return y_index_of_largest_pivot
def _compute_jordan_chains(A: np.ndarray, λ: int|float, algebraic_multiplicity: int, verbose: bool = False) -> list[np.ndarray] | None: # use for λ with algebraic multiplicity > 1 
    #diagonalize(A,True) # notify if the matrix is not diagonalizable
    #Step1: Find m, k, and r_max where m: geometric multiplicity, k: algebraic multiplicity, r_max: possible maximum size of a chain
    #  m = dim(nullspace(A - eigval*I))
    #  k is available from the input
    # r_max = k-m+1 (based on the theorem of maximum size of a chain)
    N = A - λ*np.identity(len(A)) # N = (A - λI)
    m = len(N) - np.linalg.matrix_rank(N) # based on the rank-nullity theorem
    k = algebraic_multiplicity
    r_max = k - m + 1
    #Step3: find basis for nullspace of (A - eigval*I) = N
    #always guaranteed to have at least one null space vector
    if verbose:
        print("N is: ")
        print_matrix(N)
        print(f"rank of N is: {np.linalg.matrix_rank(N)}")
    null_space_vects = compute_nullspace(N) 
    #Step4: initialize the jordan_chains with form[np.([[1,4], np([[-1],  ....
    #                                                   [2,5],      [7],   ....
    #                                                   [3,6]]),    [3]]), ....]
    # since null_space_vects contain regular eigenvectors, so let each of them be an element of the jordan_chains
    jordan_chains = []
    for null_v in null_space_vects.T:
        null_v = null_v.reshape(len(A),1)
        jordan_chains.append(null_v)
    if verbose:
      print("jordan_chains after adding the regular eigenvectors is: ")
      for i in range(len(jordan_chains)):
        print(f"the {i+1}th chain is: ")
        print_matrix(jordan_chains[i])

    #REMINDER: My faith is put on this since I know the theorems but have not been able to prove it yet, everything from step 5 to 6 is faith :)) #link to theorems:https://www.uio.no/studier/emner/matnat/math/MAT2440/v11/undervisningsmateriale/genvectors.pdf
             # the process is kind of reversed from traditional way of finding jordan chains where we start from v_lowest to v_highest possible: that is given a v1, find v2 by (A-λI).v2 = v1, then find v3 by (A-λI).v3 = v2, and so on
             #but here we go from selecting all possibl v_highest (the v's that are nullspace basis of (A - λ)^r_max ) from null_space_vects, then find v_lower till v_lowest (v1) by just by applying (A-λI) to the v_highest repeatedly, this seems eaiser but counter-intuitive
             #one advantage of this approach is that the order of P allows me to directly find a complete independent solution set of the non-diagonalizable systems of differential equations, I have proof for this
    # at this step, there are k-m vectors left 
    # the sum of those vectors ( individual vectors of each element of each column of the jordan_chains ) = k-m
    # at this stage what in jordan_chains are just regular eigenvectors
    # and we do not need null_space_vects anymore, we can change it
    #Step5: get null_space_vects of (A - λ)^r_max, based on the theorem, head of the chain (vr = vk) always satisfies (A - λ)^r_max .vr = 0
    null_space_vects = compute_nullspace(np.linalg.matrix_power(N, r_max)) 
    print("null_space_vects is: ",null_space_vects)
    #Step6: ... havenot thought of name yet but key idea is to find null_v that is LI with the col vectors in jordan_chains until there are no more null_v
    for null_v in null_space_vects.T:
        null_v = null_v.reshape(len(A),1) 
        #null_v is like np.array([[1.]
        #                         [0.]
        #                        [0.]])
        #IMPORTANT PART: check if null_v is linearly dependent with the col vectors in jordan_chains, else continue to next null_v
        skip_this_null_v = False #flag
        for chain in jordan_chains:
            if _check_LD_between_matrix_and_vector(chain, null_v):
                skip_this_null_v = True
                break
        if skip_this_null_v == True:
            continue
        #else continue to next null_v
        if verbose:
           print("this LI null_v is: ")
           print_matrix(null_v)

        #now make a new chain and add to jordan_chains
        new_chain = copy.deepcopy(null_v) # new chain is just a colvector
        while True: # given vk, find vk-1, vk-2,... v1 ( new_chain = np([v1,v2,...,vk]) 
                    # and v1 is where v0 = 0 if repeatedly mulplty by (A - λ) = N
            null_v = np.dot(N,null_v)
            #IMPORTANT PART: iterate through each element of jordan_chains to check if null_v is linearly dependent with the col vectors in jordan_chains
                             #if true, then remove the element from jordan_chains, then everything is ok

            if np.all(null_v == 0): # check if null_v is a zero column vector
                break # stop
            #else 
            new_chain = np.column_stack( (null_v,new_chain) ) #vk, vk-1,...v1 as a matrix
        
        #given a new_chain, check linear independence with each col vector in each chain in jordan_chains to a new_chain
            index_of_a_chain_to_be_deleted = 0 #starting index
            for chain in jordan_chains: #this loop is for checking linear indepdendence
                for col_vect in chain.T: #iterate through each col vector in a chain
                    col_vect = col_vect.reshape(len(A),1)
                    if _check_LD_between_matrix_and_vector(new_chain, col_vect): #true if linearly dependent
                        if verbose:
                           print(f"we delete the {index_of_a_chain_to_be_deleted+1}th chain in {len(jordan_chains)} chains")
                           print("that chain before deleting is: ")
                           print_matrix(jordan_chains[index_of_a_chain_to_be_deleted])
                        jordan_chains.pop(index_of_a_chain_to_be_deleted)  
                        break # stop poping, we have found the chain to be deleted
                index_of_a_chain_to_be_deleted += 1
        
        #then add the new_chain to the top of jordan_chains
        if verbose:
            print(f"because length of new_chain is: {len(new_chain[0])}")
            print("new_chain looks like: ")
            print_matrix(new_chain)
        jordan_chains.insert(0,new_chain) # add to top of list (1st index)
        #break
        
    
    #lastly, check if total num of col vectors in each chain in jordan_chains > k,
    #  if it is, then pop the a chain that has only 1 col vector
    sum_of_col_vects = 0
    for i in range(len(jordan_chains)):
        sum_of_col_vects += len(jordan_chains[i][0])
    if sum_of_col_vects > k:
        for i in range(len(jordan_chains)):
            if len(jordan_chains[i][0]) == 1:
                jordan_chains.pop(i)
                break
    if verbose:
        for i in range(len(jordan_chains)):
            print(f"the {i}th chain is: ")
            print_matrix(jordan_chains[i])
    return jordan_chains

def _check_LD_between_matrix_and_vector(matrix: np.ndarray, vector: np.ndarray, verbose: bool = False) -> bool:
    rank = np.linalg.matrix_rank( np.column_stack(  (matrix, vector)  ))
    if rank == len(matrix[0]):
        return True
    else:
        return False

def print_matrix(matrix: np.ndarray) -> None:
    print("////////////////////////////////////////")
    #round numpy array to 3 decimal places for better readability but not actually cutting off the values
    matrix = np.round(matrix,3)
    for row in matrix:
        print(row)
    return

if __name__ == "__main__":
    main()

''' non-diagonalizable matrix
    A1 = np.array([[10,1,-2]
                ,[0,9,3],
                  [-1,-1,8]])
    A2 = np.array([[1,1,1],
                  [0,1,0],
                  [0,0,1]])
    A3 = np.array([[12,16,-4],
                  [-5,30,-2],
                  [5,-8,24]])
    A4 = np.array([[-7,8,2],
                  [-4,5,1],
                  [-23,21,7]])
    A5 = np.array([[2,1,0],
                  [-1,0,0],
                  [0,0,1]])
    A6 = np.array([[2,1,0],
                  [0,2,0],
                  [0,0,2]])
    A7 = np.array([[2,0,1,4],
                  [0,2,0,4],
                  [0,0,2,5],
                  [0,23,0,2]])
    A8 = np.array([[2,0,1,4,6],
                   [0,2,0,-4,6],
                   [0,0,2,55,6],
                   [0,4,0,2,64],
                   [1,0,0,0,2]])
    A9 = np.array([[1,1,1],
                   [1,-21,235],
                   [1,235,1]])
'''    