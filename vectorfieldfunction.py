import numpy as np
import sympy as sp
from multiprocessing import Pool  
from multivariablefunction import MultivariableFunction
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z = sp.symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')


def main():
    #feel free to play with the class
    #test find_best_local_extrema_within_hyperspherical_bound
    f1 = MultivariableFunction([x, y, z], ((1-(6*x+16*y+z))**2+(-1-(-6*x+3*y+z))**2+(-1-(3*x-6*y+z))**2+(-1-(-3*x+3*y+z))**2)/4)
    print(f1.find_best_local_extrema_within_hyperspherical_bound("minima", step_size = 0.001, 
                                                                 norm_tolerance = 0.01, 
                                                                 hypershperical_constraint = 20, 
                                                                 density_1D = 3, 
                                                                 num_of_cores = 3,
                                                                 verbose = True))


class VectorFieldFunction:
    def __init__(self, independent_vars: list[sp.Symbol], expr: list[sp.Expr]):
        self.independent_vars = independent_vars
        self.expr = expr
    @staticmethod
    def vector_surface_integral(F: "VectorFieldFunction", r: "VectorFieldFunction", bounds: list[tuple[float, float]]) -> int|float:
        '''
        Given r( x(u,v), y(u,v), z(u,v) ) = r(u,v) as a vector field function for the purpose of surface parametrization
        Given F(x,y,z) as a vector function
        Formula: ∫∫_s f•N ds = ∫∫_D F(r_x(u,v), r_y(u,v), r_z(u,v)) • (∂r/∂u x ∂r/∂v) dudv
        N is the unit normal vector = ∂r/∂u x ∂r/∂v / || ∂r/∂u x ∂r/∂v ||
        S is surface area parametrized by r(u,v)
        '''
        #step 1: compute the cross product of ∂r/∂u and ∂r/∂v
        dif_r_with_u = r.diff(r.independent_vars[0])
        print(f"dif_r_with_u: {dif_r_with_u.expr}")
        dif_r_with_v = r.diff(r.independent_vars[1])
        print(f"dif_r_with_v: {dif_r_with_v.expr}")
        cross_product = dif_r_with_u.cross(dif_r_with_v)    
        #step 3: sub x = r_x(u,v), y = r_y(u,v), z = r_z(u,v) into F(x,y,z) to get F(u,v)
        vector_func = VectorFieldFunction(r.independent_vars, [F.expr[i].subs({F.independent_vars[j]: r.expr[j] for j in range(len(F.independent_vars))}) for i in range(len(F.expr))])
        #step 4: compute F(u,v) • (∂r/∂u x ∂r/∂v) and integrate over the domain
        print(f"cross_product: {cross_product.expr}")
        print(f"vector_func: {vector_func.expr}")
        final_form = vector_func.dot(cross_product)
        return final_form.multivar_integral(bounds)

    def surface_area(self, bounds : list[tuple[float, float]]): #used when 2d surface is parametrized
        '''
        Given r( x(u,v), y(u,v), z(u,v) ) = r(u,v), the surface area of the graph of r is given by the formula:
        Formula: ∫∫_D || ∂r/∂u x ∂r/∂v || dudv
        S is surface area parametrized by r(u,v)
        '''
        #step 1: compute the cross product of ∂r/∂u andand ∂r/∂v
        dif_r_with_u = self.diff(self.independent_vars[0])
        dif_r_with_v = self.diff(self.independent_vars[1])
        cross_product = dif_r_with_u.cross(dif_r_with_v)
        #step 2: compute the norm of the cross product
        norm_cross_product = cross_product.norm() #this gives a multivariable function
        #step 3: integrate the norm of the cross product over the domain
        norm_cross_product.multivar_integral(bounds)
        return norm_cross_product

    def scalar_surface_integral(self, scalar_func: "MultivariableFunction", bounds: list[tuple[float, float]]) -> int|float:
        '''
        Given r( x(u,v), y(u,v), z(u,v) ) = r(u,v) as a vector field function for the purpose of surface parametrization
        Given f(x,y,z) as a scalar function
        Formula: ∫∫_s f(x,y,z) ds = ∫∫_D f(r_x(u,v), r_y(u,v), r_z(u,v)) || ∂r/∂u x ∂r/∂v || dudv
        S is surface area parametrized by r(u,v)
        '''
        #step 1: compute the cross product of ∂r/∂u and ∂r/∂v
        dif_r_with_u = self.diff(self.independent_vars[0])
        dif_r_with_v = self.diff(self.independent_vars[1])
        cross_product = dif_r_with_u.cross(dif_r_with_v)
        #step 2: compute the norm of the cross product
        norm_cross_product = cross_product.norm() #this gives a multivariable function || ∂r/∂u x ∂r/∂v ||
        #step 3: sub x = r_x(u,v), y = r_y(u,v), z = r_z(u,v) into f(x,y,z) to get f(u,v)
        scalar_func = MultivariableFunction( [self.independent_vars[0], self.independent_vars[1]], scalar_func.expr.subs({scalar_func.independent_vars[i]: self.expr[i] for i in range(len(self.independent_vars))}) )
        #step 4: multiply f(u,v) and || ∂r/∂u x ∂r/∂v || and integrate over the domain
        final_form = MultivariableFunction(self.independent_vars, scalar_func.expr * norm_cross_product.expr)
        return final_form.multivar_integral(bounds)

    def vector_line_integral(self, r: "VectorFieldFunction", start_time: float, end_time: float, verbose = False)-> int|float:
        '''
        Formula: ∫f(t)•r'(t)dt from a to b
        where f(t) and r'(t) are the vector function (vvf = r)
        for example, f(x,y,z) = VectorFieldFunction([x,y,z], [x**2, y**2, z**2]) 
        and r(t) =  VectorFieldFunction([x(t), y(t), z(t)])
        '''
        #Step 1: sub x = x(t), y = y(t), z = z(t) into f(x,y,z) to get f(t)
        f = VectorFieldFunction(  r.independent_vars , [self.expr[i].subs({self.independent_vars[j]: r.expr[j] for j in range(len(self.independent_vars))}) for i in range(len(self.expr))] )
        #Step 2: compute r'(t)
        r_prime = r.diff(t)
        print(r_prime.expr)
        #Step 3: compute the dot product of f(t) and r'(t) to get f(t)•r'(t)
        if verbose:
            print(f'f(t)={f.expr}')
            print(f'r\'(t)={r_prime.expr}')
        return sp.integrate(f.dot(r_prime).expr, (*r.independent_vars, start_time, end_time))
     
    def vector_flux_integral(self, var_for_r: sp.Symbol, r: "VectorFieldFunction", start_time: float, end_time: float)-> int|float:
        '''
        formula: ∫  [f(t).N(t)] ||r'(t)||dt from a to b 
        where f(t) and N(t) are the vector function
        N(t) is the unit normal vector of r(t)
        '''
        #Step 1: sub x = x(t), y = y(t), z = z(t) into f(x,y,z) to get f(t)
        f = VectorFieldFunction(  r.independent_vars , [self.expr[i].subs({self.independent_vars[j]: r.expr[j] for j in range(len(self.independent_vars))}) for i in range(len(self.expr))] )
        #Step 2: compute N(t) 
        N = r.unit_normal_v(var_for_r)
        #step 3: compute ||r'(t)|| 
        r_prime = r.diff(var_for_r).norm()
        #Step 4: get f(t)•N(t) ||r'(t)|| and integrate from a to b with respect to t
        return sp.integrate( f.dot(N).expr * r_prime.expr, (var_for_r, start_time, end_time))
      
    def unit_tangent_v(self, var: sp.Symbol, verbose=False)-> "VectorFieldFunction":
        r_1st_diff = self.diff(var)
        norm_r_1st_diff = r_1st_diff.norm()
        T = r_1st_diff.scale(1/norm_r_1st_diff.expr)
        if verbose:
            print("T:", T.expr)
        return T

    def unit_normal_v(self, var: sp.Symbol, verbose=False)-> "VectorFieldFunction":
        T = self.unit_tangent_v(var)
        T_1st_diff = T.diff(var)
        norm_T_1st_diff = T_1st_diff.norm()
        N = T_1st_diff.scale(1/norm_T_1st_diff.expr)
        if verbose:
            print("N:", N.expr)
        return N

    def unit_binormal_v(self, var: sp.Symbol, verbose=False)-> "VectorFieldFunction":
        T = self.unit_tangent_v(var)
        N = self.unit_normal_v(var)
        B = T.cross(N)
        if verbose:
            print("B:", B.expr)
        return B 

    def norm(self):
        return MultivariableFunction(self.independent_vars ,sp.sqrt(sum([v**2 for v in self.expr])))

    def diff(self, var: sp.Symbol):
        return VectorFieldFunction(self.independent_vars, [sp.diff(v, var) for v in self.expr])

    def integrate(self, var: sp.Symbol): 
        return VectorFieldFunction(self.independent_vars,[sp.integrate(v, var) for v in self.expr])

    def add(self, vvf2: list[sp.Expr]) :
        return VectorFieldFunction(self.independent_vars, [v1 + v2 for v1, v2 in zip(self.expr, vvf2)])

    def subtract(self, vvf2: list[sp.Expr]) :
        return VectorFieldFunction(self.independent_vars, [v1 - v2 for v1, v2 in zip(self.expr, vvf2)])

    def scale(self, scalar: sp.Expr) :
        return VectorFieldFunction(self.independent_vars, [v * scalar for v in self.expr])

    def multiply(self, vvf2: "VectorFieldFunction"):
        return VectorFieldFunction(self.independent_vars, [v1 * v2 for v1, v2 in zip(self.expr, vvf2.expr)])

    def divide(self, vvf2: "VectorFieldFunction") :
        return VectorFieldFunction(self.independent_vars, [v1 / v2 for v1, v2 in zip(self.expr, vvf2.expr)])

    def dot(self, vvf2: "VectorFieldFunction") :
        return MultivariableFunction(self.independent_vars, sum([v1 * v2 for v1, v2 in zip(self.expr, vvf2.expr)]))

    def cross(self, vvf2: "VectorFieldFunction") :
        if len(self.expr) == 3 and len(vvf2.expr) == 3:
            return VectorFieldFunction(self.independent_vars + vvf2.independent_vars ,[
                self.expr[1] * vvf2.expr[2] - self.expr[2] * vvf2.expr[1],
                self.expr[2] * vvf2.expr[0] - self.expr[0] * vvf2.expr[2],
                self.expr[0] * vvf2.expr[1] - self.expr[1] * vvf2.expr[0]
            ])
        else:
            raise ValueError("Cross product is defined only for 3-dimensional vectors.")

if __name__ == "__main__":
    main()
