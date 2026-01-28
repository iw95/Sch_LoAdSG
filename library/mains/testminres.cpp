#include <sys/time.h>
#include <cstdlib>
#include "../source/applications/minres.h"




int main(int argc, char **argv) {
    srand(time({})); // use current time as seed for random generator

    cout << "Level\tErrorP\tIterationsP\t\tErrorH\tIterationsH" << endl;


    int level_start=1;

    AdaptiveSparseGrid grid;
    IndexDimension centerPoint;

    double Linfty_old = 1.0;


    for (int level = level_start; level <9; level++)
    // for (int level = 2; level < 3; level++)
    {
        // Start measuring time
        struct timeval begin_all, end_all;
        gettimeofday(&begin_all, 0);

        IndexDimension centerPoint;
        grid.AddRecursiveSonsOfPoint(centerPoint,level);

        MultiLevelAdaptiveSparseGrid mgrid(&grid);
        MatrixVectorHomogen m(grid, mgrid, 1, 0);


        Poisson poisson(grid);
        HelmHoltz helmholtz(grid);
        VectorSparseG x(grid);
        VectorSparseG b(grid);
        VectorSparseG bH(grid);
        VectorSparseG xminres(grid);
        VectorSparseG xdiff(grid);
        VectorSparseG x0(grid);
        //x = 1.;
        // b = 1.;
        // calcPrewByNodal(x, b);
        for (unsigned long i = 0; i < grid.getMaximalOccupiedSecondTable(); i++) {
                IndexDimension I = grid.getIndexOfTable(i);
                double valx = rand() / RAND_MAX;
                x.setValue(i,valx);
        }
        x0 = x;

        m.multiplication(x, b, poisson);
        m.multiplication(x, bH, helmholtz);

        // change x to not be solution
        for (unsigned long i = 0; i < grid.getMaximalOccupiedSecondTable(); i++) {
                IndexDimension I = grid.getIndexOfTable(i);
                double valx = rand() / double(RAND_MAX);
                x0.setValue(i,valx);
        }
        

        int minres_iterations;
        int minres_iterH;
        double precontime;

        MinRes::solveHomogen<Poisson>(1e-10, x0, b, &minres_iterations, xminres, m, poisson, &precontime);
        xdiff = x - xminres;
        double error = product(xdiff, xdiff);

        MinRes::solveHomogen<HelmHoltz>(1e-10, x0, bH, &minres_iterH, xminres, m, helmholtz, &precontime);
        xdiff = x - xminres;
        double errorH = product(xdiff, xdiff);


        cout << level << "\t" << error << "\t" << minres_iterations;
        cout << "\t\t" << errorH << "\t" << minres_iterH << endl;

    }

    return 0;


}