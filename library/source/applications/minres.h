//
// Created by rumi on 21.01.26.
//


// TODO needed?
#ifndef GRUN_MinRes_METHOD_H
#define GRUN_MinRes_METHOD_H


#include <sys/time.h>
#include <unordered_set>
#include "../MatrixVectorMultiplication/MatrixVectorHomogen.h"
#include "../MatrixVectorMultiplication/MatrixVectorInhomogen.h"

class MinRes {
public:
    template<class Stencil>
    static bool solveHomogen(double eps,
                             VectorSparseG &x0,
                             VectorSparseG &f,
                             int *iterations,
                             VectorSparseG &usol,
                             MatrixVectorHomogen& matrix,Stencil stencil,double* time_precon);

    static const int maxIteration = 500;
    static double error[MaximumDepth][maxIteration];
    static int count;


    static void printError() {
        ofstream Datei;
        Datei.open("../results/error_MinRes.gnu", std::ios::out);

        for (int i = 0; i < MaximumDepth; i++) {
            if (error[i][0] != 0) {
                Datei << "\t" << error[i][0];
                cout << "\t" << error[i][0];
            }
        }
        Datei << endl;
        cout << endl;
        for (int i = 1; i < maxIteration; i++) {
            Datei << i;
            cout << i;
            for (int j = 0; j < MaximumDepth; j++) {
                if (error[j][0] != 0) {
                    Datei << "\t" << error[j][i];
                    cout << "\t " << error[j][i];
                }


            }

            Datei << "\n";
            cout << "\n";

        }
        Datei << endl;
        Datei.close();
    }
};




//INLINE TEMPLATE FUNCTIONS
template<class Stencil>
bool
MinRes::solveHomogen(double eps, VectorSparseG &x0,VectorSparseG &f,int *iterations,
                 VectorSparseG &usol, MatrixVectorHomogen& matrix,Stencil stencil,double* time_precon) {

    int k = maxIteration;
    double eps_sq = eps * eps;

    AdaptiveSparseGrid* grid = (x0.getSparseGrid());
    ListOfDepthOrderedSubgrids list(*grid);

    if(!(x0.getSparseGrid()->getKey()==f.getSparseGrid()->getKey()))exit(1);
    
    

    // Fields
    VectorSparseG x_iter(grid);
    x_iter = x0;

    VectorSparseG r(grid);
    VectorSparseG p2(grid);
    VectorSparseG p1(grid);
    VectorSparseG p0(grid);
    VectorSparseG s2(grid);
    VectorSparseG s1(grid);
    VectorSparseG s0(grid);

    VectorSparseG z(grid);

    double alpha;
    double beta1, beta2;

    
    // solves  Ax = f;


    // r = A*x - f;

    // Start measuring time
    struct timeval begin_all, end_all;
    gettimeofday(&begin_all, 0);



// PRECONDITIONING
    // Start measuring time
    // struct timeval begin, end;
    // gettimeofday(&begin, 0);
    // Preconditioning<Stencil> P(z,stencil);

    // // Stop measuring time and calculate the elapsed time
    // gettimeofday(&end, 0);
    // long seconds = end.tv_sec - begin.tv_sec;
    // long microseconds = end.tv_usec - begin.tv_usec;
    // double duration_def = seconds + microseconds*1e-6;
    // *time_precon = duration_def;
//


    

    // r = Ax - f
    matrix.multiplication<Stencil>(x_iter, r, stencil);
    // cout << "Ax\t" << product(r,r) << endl;
    r = f - r;

    // cout << "xiter\t" << product(x_iter, x_iter) << endl;
    // cout << "r\t" << product(r,r) << endl;

    // pi = r
    // si = Api
    p0 = r;
    matrix.multiplication<Stencil>(p0, s0, stencil);

    if (product(s0,s0) == 0) {
        usol = x0;
        *iterations = 0;
        return false;
    }

    // cout << "p0\t" << product(p0,p0) << endl;
    // cout << "s0\t" << product(s0,s0) << endl;


    for (int i = 1; i <= maxIteration; ++i) {
        // grid->WorkOnHangingNodes = false;

        p2 = p1; p1 = p0;
        s2 = s1; s1 = s0;

        // changed s0 to s1
        alpha = product(r, s1) / product(s1, s1);

        x_iter = x_iter + alpha * p1;

        r = r - alpha * s1;

        // cout << "alpha\t" << alpha << endl;
        // cout << "xiter\t" << product(x_iter,x_iter) << endl;
        // cout << "r\t" << product(r,r) << endl;


        double resid = product(r,r);
        // cout << resid << endl;
        if (resid < eps_sq || isnan(resid)) {
            k = i;
            break;
        }


        // pi = s_{i-1}
        // si = A s_{i-1}
        p0 = s1;
        matrix.multiplication<Stencil>(s1, s0, stencil);


        beta1 = product(s0, s1) / product(s1, s1);

        p0 = p0 - beta1 * p1;
        s0 = s0 - beta1 * s1;


        if (i > 1) {
            beta2 = product(s0, s2) / product(s2, s2);

            p0 = p0 - beta2 * p2;
            s0 = s0 - beta2 * s2;

        }



    }


    //cout << "MinRes finished after " << k << " iterations and residuum delta = " << sqrt(product(r,r)) << endl;
    usol = x_iter;
    *iterations = k;
    return true;
}

// needed?
#endif //GRUN_MinRes_METHOD_H
