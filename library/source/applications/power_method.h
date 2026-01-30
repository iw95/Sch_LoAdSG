//
// Created by rumi on 19.12.24
//

#ifndef GRUN_Power_METHOD_H
#define GRUN_Power_METHOD_H


#include <sys/time.h>
#include <unordered_set>
#include "../MatrixVectorMultiplication/MatrixVectorHomogen.h"
#include "../MatrixVectorMultiplication/MatrixVectorInhomogen.h"
#include "cg_method.h"
#include "minres.h"

class Power {
public:

    template<class Stencil_left, class Stencil_right>
    static bool power_max(double eps, VectorSparseG &x,
                            double &eigenvalue, int &iterations,
                            MatrixVectorHomogen& matrix,Stencil_left lhs, Stencil_right rhs,
                            bool rightoperator, double &precondition, double &duration);

    template<class Stencil_left, class Stencil_right>
    static bool power_inverse(double eps, VectorSparseG &x,
                            double &eigenvalue, int &iterations,
                            double &avg_cg_iterations, MatrixVectorHomogen& matrix,
                            Stencil_left& lhs, Stencil_right& rhs,
                            bool rightoperator, double &precondition, double &duration);

    template<class Stencil_left, class Stencil_right>
    static double eigenvalue_exp(MatrixVectorHomogen &matrix, Stencil_left &lhs, Stencil_right &rhs, VectorSparseG &u, AdaptiveSparseGrid &grid);
   
    

    // template<class Stencil>
    // double precon(Stencil stencil, AdaptiveSparseGrid* grid);

    static const int maxIteration = 500;
    static double error[MaximumDepth][maxIteration];
    static int count;

template<class Stencil>
static double precon(VectorSparseG z, Stencil stencil);

    
};


template<class Stencil_left, class Stencil_right>
double 
Power::eigenvalue_exp(MatrixVectorHomogen &matrix, Stencil_left &lhs, Stencil_right &rhs, VectorSparseG &u, AdaptiveSparseGrid &grid) {

    VectorSparseG left(grid), right(grid);
    matrix.multiplication<Stencil_left>(u, left, lhs);
    matrix.multiplication<Stencil_right>(u, right, rhs);

    double scal_left = sqrt(product(left,left));
    double scal_right = sqrt(product(right,right));
    return scal_left / scal_right;
}



template<class Stencil>
double Power::precon(VectorSparseG z, Stencil stencil) {

    // Start measuring time of preconditioning
    struct timeval begin_pre, end_pre;
    gettimeofday(&begin_pre, 0);

    //dirichlet_grid.completeDirichletGrid();  ???
    Preconditioning<Stencil> P(z,stencil);

    gettimeofday(&end_pre, 0);
    long pre_seconds = end_pre.tv_sec - begin_pre.tv_sec;
    long pre_microseconds = end_pre.tv_usec - begin_pre.tv_usec;
    double pre_duration_def = pre_seconds + pre_microseconds*1e-6;
    
    return pre_duration_def;
}



//INLINE TEMPLATE FUNCTIONS
template<class Stencil_left, class Stencil_right>
bool
Power::power_max(double eps, VectorSparseG &x, double &eigenvalue, int &iterations,
                 MatrixVectorHomogen& matrix, Stencil_left lhs, Stencil_right rhs, bool rightoperator, double &precondition, double &duration) {


    AdaptiveSparseGrid* grid = (x.getSparseGrid());
    ListOfDepthOrderedSubgrids list(*grid);

    VectorSparseG z(grid);
    VectorSparseG xalt(grid);
    VectorSparseG y(grid);
    VectorSparseG xneu(grid);
    xalt = x / sqrt(product(x,x));

    // solves  Ax = M * (lambda * x); for lambda and x


    // Preconditioning
    if (precondition != 0) {
        precondition = Power::precon<Stencil_right>(z, rhs);
    }


    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);


    // norm of vector y
    double norm_x = 0;
    double norm_x_old = 0;


    int k = 0;

    // parameters for cg iteration to solve rhs
    int cg_iterations = 0;
    double cg_time = 0.0;
    double cg_eps = eps * 1e-2;
    
    // cout << endl;

    // Iterate until maxiter
    for (int i = 0; i < maxIteration; i++) {
        // Remember iterations
        k = i;

        // y := A * xalt
        matrix.multiplication<Stencil_left>(xalt, y, lhs);
        if (rightoperator) {
            // solve (y = M * x_neu) for xneu
            CG::solveHomogen<Stencil_right>(cg_eps, xneu, y, &cg_iterations, xalt, matrix, rhs, &cg_time);
        } else {
            xneu = y;
        }

        norm_x = sqrt(product(xneu, xneu)); /// grid->getDOFS());

        if (abs(norm_x - norm_x_old) < eps)
            break;
        

        // Update norm_old and xalt
        norm_x_old = norm_x;

        // normalize vector x := xneu / |xneu|
        xalt = xneu / norm_x;

    }


    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    duration = seconds + microseconds*1e-6;


    eigenvalue = norm_x;
    iterations = k;
    x = xneu/norm_x;

    return true;
}



//INLINE TEMPLATE FUNCTIONS
template<class Stencil_left, class Stencil_right>
bool
Power::power_inverse(double eps, VectorSparseG &x, double &eigenvalue, int &iterations, double &avg_cg_iterations,
                 MatrixVectorHomogen& matrix, Stencil_left& lhs, Stencil_right& rhs, bool rightoperator, double &precondition, double &duration) {



    AdaptiveSparseGrid* grid = (x.getSparseGrid());
    ListOfDepthOrderedSubgrids list(*grid);

    VectorSparseG z(grid);
    VectorSparseG xalt(grid);
    VectorSparseG y(grid);
    VectorSparseG xneu(grid);
    xalt = x;


    // solves  Ax = M * (lambda * x); for smallest lambda and x

    // Preconditioning
    if (precondition != 0) {
        precondition = Power::precon<Stencil_left>(z, lhs);
    }

    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);


    // seminorm of vector xneu
    double norm_x = 0;
    double norm_x_old = 0;

    int k = 0;

    // (return) parameters for cg iteration to solve rhs
    int cg_iterations = 0;
    double cg_time = 0.0;
    double cg_eps = eps * 1e-2;
    double avg_cg_iter = 0;
    

    // Iterate until maxiter
    for (int i = 0; i < maxIteration; i++) {
        // Remember iterations
        k = i;

        if (rightoperator) {
            // y := M * xalt
            matrix.multiplication<Stencil_right>(xalt, y, rhs);
        } else {
            y = xalt;
        }
        // solve (y = A * x_neu) for xneu
        CG::solveHomogen<Stencil_left>(cg_eps, xneu, y, &cg_iterations, xalt, matrix, lhs, &cg_time);


        norm_x = sqrt(product(xneu, xneu));

        if (abs(norm_x - norm_x_old) < eps)
            break;

        
        norm_x_old = norm_x;
        xalt = xneu / norm_x;

        avg_cg_iter += cg_iterations;
    }

    avg_cg_iter /= k;


    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    duration = seconds + microseconds*1e-6;


    eigenvalue = 1 / norm_x;
    iterations = k;
    avg_cg_iterations = avg_cg_iter;
    x = xneu/norm_x;

    return true;
}



#endif //GRUN_CG_METHOD_H
