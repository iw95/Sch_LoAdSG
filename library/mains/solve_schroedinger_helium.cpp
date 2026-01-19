//
// Created by rumi on 19/01/26.
//

#include "../source/sparseEXPDE.h"
#include "../source/iterator/depthIterator.h"
#include "../source/applications/power_method.h"
//#define testNOW
//#define goolePerf
#include <string>
#include <ctime>
#include <sys/time.h>



double border = 7.5;
//3e-9;
//8e-10;

double eps = 1e-35;

const double CUTCOEFF = 1e10;

const int mc_samples = 100;


int num_electrons = 2;

    // https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
    double bohr_radius = 5.29177210544e-11;
    double b_factor = sqrt(M_PI);

    // TODO check varc factor
    const double varc_factor = -8*border;

    const double rhs_factor = 4*border*border;

    // modular for num_electrons
    const double eig_exp = -1.*rhs_factor;


double alpha = 1.;

bool trick = false;


void mpi_cout(string s, bool endline = true){
    int rank = 0;
#ifdef MY_MPI_ON
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank ==0){
        cout << s;
        if (endline) cout << endl;
    }
#else
        cout << s;
        if (endline) cout << endl;
#endif
}





double electron_core(double* coordinates, size_t electron_idx) {
    double r = 0;
    for (int i = DimensionSparseGrid*electron_idx; i < DimensionSparseGrid*(electron_idx+1); i++) {
        double ci = 2. *  coordinates[i] - 1; // < map from [0,1] to [-s,s]
        r += ci * ci;
    }
    r = sqrt(r);

    double result  = CUTCOEFF;
    if (r!=0) {
        result = min(1./r, CUTCOEFF);
    }

    return result;
    // Still to happen in : varc_factor for coordinate transformation
    // trick for positive definiteness
    // electron charge?
}


double electron_electron(double* coordinates, size_t e_idx0, size_t e_idx1) {
    double r = 0;
    int offset0 = DimensionSparseGrid*e_idx0;
    int offset1 = DimensionSparseGrid*e_idx1;
    for(int i = 0; i < DimensionSparseGrid; i++) {
        double ci = coordinates[offset0 + i] - coordinates[offset1 +  i];
        r += ci*ci;
    }
    r = 2*sqrt(r);

    double result = CUTCOEFF;
    if (r != 0) {
        result = min(1./r, CUTCOEFF);
    }

    return result;
    // Still to happen: varc_factor for coordinate transformation
    // trick for positive definiteness
    // core and electron charge?
}


double var_coeff( double* coordinates)
{
    double result = 0.;

    for (int e = 0; e < num_electrons; e++) {
        // Electron - core interaction term
        result -= electron_core(coordinates, e);

        for (int e2 = 0; e2 < e; e2++) {
            // Electron - electron interaction term
            result += electron_electron(coordinates, e, e2);
        }
    }

    // Factor from coordinate transformation
    result *= varc_factor;

    // Trick to make positive definite
    if (trick) {
        result -= 1.9 * eig_exp;
    }

    return result * alpha;
}




int main(int argc, char **argv) {

    int rank = 0;
    int num_tasks = 1;


    #ifdef MY_MPI_ON
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    #endif



    mpi_cout(" Dimension " + to_string( DimensionSparseGrid));
    mpi_cout("Regular grid.");


    #pragma omp parallel
        {
    #pragma omp single
            {
                int total_threads = omp_get_num_threads();
                mpi_cout("number of mpi_tasks " + to_string(num_tasks));
                mpi_cout("number of omp_threads " + to_string(total_threads));
            }
        }


    int numberMVprocesses=1;
    int numberLSprocesses=num_tasks - numberMVprocesses;

    mpi_cout("use " + to_string(numberLSprocesses)+ " for LocalStiffnessmatrices ");
    mpi_cout("use " + to_string(num_tasks - numberLSprocesses)+ " for Matrix Vector Multiplication ");





    if (rank == 0) {
        cout << "Computing the first eigenvalue of the Schroedinger equation for the HELIUM atom, i.e. with (originally) two electrons." << endl;
        cout << "The expected eigenvalue is " << "?" << endl; // TODO
        cout << "Computing for " << num_electrons << " electrons." << endl << endl;
        cout << "Computing on domain [-" << border << ", " << border << "]" << endl;
        cout << "With epsilon = " << eps << endl;
        cout << "And Coefficient cutoff = " << CUTCOEFF << endl;
        cout << "Monte Carlo samples = " << mc_samples << endl << endl;

        //cout << "Factor k^2 = " << varc_factor << endl << endl;

        string legend = "level\tDOFS\t\talph=";
        for (double alpha_local = 0.; alpha_local <= 1; alpha_local += 0.1) {
            legend += "\t" + to_string(alpha_local);
        }
        legend += "\t\tcg_iter";
        cout << legend << endl << endl;
    }







    int level_start=1;

    AdaptiveSparseGrid grid;
    IndexDimension centerPoint;


    double Linfty_old = 1.0;

    for (int level = level_start; level <9; level++)
    {
        // Start measuring time
        struct timeval begin_all, end_all;
        gettimeofday(&begin_all, 0);

        IndexDimension centerPoint;
        grid.AddRecursiveSonsOfPoint(centerPoint,level);

        MultiLevelAdaptiveSparseGrid mgrid(&grid);
        MatrixVectorHomogen m(grid, mgrid, 1, 0);

        Poisson poisson(grid);



        string powers_str = "";
        
        double alphasteps = 10;
        for (int alpha_local = 0; alpha_local <= alphasteps; alpha_local += 1) {

            alpha = alpha_local / alphasteps;


            HelmHoltz rhs(grid);

            StencilMC<double (*)(double *)> stencilVarCoeff(grid,&var_coeff,mc_samples);
            LocalStiffnessMatricesDynamicDistribution lhs(grid, stencilVarCoeff,numberLSprocesses);
            lhs.addStencil(poisson);




            double eigenvalue_power;

            int pow_interations;
            double cg_interations;
            double time_power;
            double precon = 0;
            VectorSparseG prew_inv(grid);
            prew_inv = 1.0;

            Power::power_inverse<LocalStiffnessMatricesDynamicDistribution, HelmHoltz>(eps, prew_inv, eigenvalue_power, pow_interations, cg_interations,
                m, lhs, rhs, true, precon, time_power);


            powers_str += "\t" + to_string(eigenvalue_power/rhs_factor);

            if (alpha_local == alphasteps) {
                powers_str += "\t\t" + to_string(cg_interations);
            }
        
        }

        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\t", false);
        mpi_cout("power\t", false);
        mpi_cout(powers_str);

        mpi_cout("");

    }

#ifdef MY_MPI_ON
    MPI_Finalize();
#endif
    return 0;


}