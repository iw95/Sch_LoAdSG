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
#include <iostream>
#include <fstream>



double border = 7.5;
double eps = 1e-35;
const int lvl = 4;

const int count = 3;

const double CUTCOEFFarray[] = {1e1, 1e2, 1e5, 1e20};
double CUTCOEFF;
const int mcsamplesarray[] = {1, 50, 100};
int mc_sample;
const double shiftfactorarray[] = {0.9, 5., 100};
double shiftfactor;

const size_t singleDimension = 3;
int num_electrons = 2;

    /// @brief Factor for variable coefficient
    /// @details 2: for factor before Helmholtz operator in atomic units
    /// 4*border: for coordinate transform factor
    const double varc_factor = 2*4*border;

    /// @brief factor for scaling eigenvalue
    /// @details from coordinate transform and Helmholtz factor
    const double rhs_factor = 2*4*border*border;

    /// @brief expected eigenvalue for one core and two electrons in atomic units
    const double eig_exp = -2.90338583;
    const double transf_eig_exp = eig_exp*rhs_factor;
    double shiftvalue;

void set_shiftvalue(double newfactor) {
    shiftvalue = (newfactor+1) * (-1) * transf_eig_exp;
}


/// @brief mpi appropriate output
/// @param s output string
/// @param endline whether to end with linebreak
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




/// @brief variable coefficient term for electron of given index and a core
/// @param coordinates coordinates for function evaluation
/// @param electron_idx index of the electron in question
/// @return value at coordinates
double electron_core(double* coordinates, size_t electron_idx) {
    double r = 0;
    for (int i = singleDimension*electron_idx; i < singleDimension*(electron_idx+1); i++) {
        double ci = 2. *  coordinates[i] - 1; // < map from [0,1] to [-s,s]
        r += ci * ci;
    }
    r = sqrt(r);

    double result  = CUTCOEFF;
    if (r!=0) {
        result = min(1./r, CUTCOEFF);
    }

    return -2 * result;
    // Still to happen in : varc_factor for coordinate transformation
    // trick for positive definiteness
}


/// @brief variable coefficient term for two electrons of given indices
/// @param coordinates coordinates for function evaluation
/// @param e_idx0 index of first electron
/// @param e_idx1 index of second electron
/// @return value at coordinates
double electron_electron(double* coordinates, size_t e_idx0, size_t e_idx1) {
    double r = 0;
    int offset0 = singleDimension*e_idx0;
    int offset1 = singleDimension*e_idx1;
    for(int i = 0; i < singleDimension; i++) {
        double ci = coordinates[offset0 + i] - coordinates[offset1 +  i];
        r += ci*ci;
    }
    r = sqrt(r);

    double result = CUTCOEFF;
    if (r != 0) {
        result = min(1./r, CUTCOEFF);
    }

    return 0.5*result;
    // Still to happen: varc_factor for coordinate transformation
    // trick for positive definiteness
}


/// @brief Calculates variable coefficient for two electrons and one core
/// @param coordinates coordinates for function evaluation
/// @return value at coordinates
double var_coeff(double* coordinates)
{
    double result = 0.;

    for (int e = 0; e < num_electrons; e++) {
        // Electron - core interaction term
        result += electron_core(coordinates, e);

        for (int e2 = 0; e2 < e; e2++) {
            // Electron - electron interaction term
            result += electron_electron(coordinates, e, e2);
        }
    }

    // Factor from coordinate transformation
    result *= varc_factor;

    // Trick to make positive definite
    result += shiftvalue;

    return result;
}




int main(int argc, char **argv) {
    // writing output to a file
    freopen ("output.txt","w",stdout);
    freopen ("stderr.txt","w",stderr);

    int rank = 0;
    int num_tasks = 1;


    #ifdef MY_MPI_ON
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    #endif



    mpi_cout("Grid dimension " + to_string(DimensionSparseGrid));
    mpi_cout("Regular grid.");
    mpi_cout("");

    mpi_cout("Single particle dimension " + to_string(singleDimension));

    if (DimensionSparseGrid % singleDimension != 0)exit(1);


    #pragma omp parallel
        {
    #pragma omp single
            {
                int total_threads = omp_get_num_threads();
                mpi_cout("number of mpi_tasks " + to_string(num_tasks));
                mpi_cout("number of omp_threads " + to_string(total_threads));
            }
        }


    int numberMVprocesses=32;
    int numberLSprocesses= num_tasks - numberMVprocesses;

    mpi_cout("use " + to_string(numberLSprocesses)+ " for LocalStiffnessmatrices ");
    mpi_cout("use " + to_string(numberMVprocesses)+ " for Matrix Vector Multiplication ");





    if (rank == 0) {
        cout << endl;
        cout << "Computing the first eigenvalue of the Schroedinger equation for the HELIUM atom, i.e. with two electrons." << endl;
        cout << "The expected eigenvalue is " << eig_exp << endl;
        cout << "Computing on domain [-" << border << ", " << border << "]" << endl;
        cout << "With epsilon = " << eps << endl;
        cout << "On level l = " << lvl << endl;
        cout << "Using the shift trick for positive eigenvalue with shift factors [";
            for(double sv : shiftfactorarray) {cout << to_string(sv) << ", ";}
            cout << "]." << endl;
        cout << "Coefficient cutoffs = [";
            for (double cc : CUTCOEFFarray) {cout << to_string(cc) << ", ";}
            cout << "]." << endl;
        cout << "Monte Carlo sampless = [";
            for (int mcs : mcsamplesarray) {cout << to_string(mcs) << ", ";}
            cout << "]." << endl;

        cout << endl << endl;

        string legend = "level\tDOFS\t\tshift\tclip\tmcsampl\t\teig\t\teig_diff\t\tcg_iter\t\tpow_iter\ttime";
        cout << legend << endl << endl;
    }




    // Regular grid construction
    AdaptiveSparseGrid grid;
    IndexDimension centerPoint;
    IndexDimension centerPoint;
    grid.AddRecursiveSonsOfPoint(centerPoint,lvl);

    MultiLevelAdaptiveSparseGrid mgrid(&grid);
    MatrixVectorHomogen m(grid, mgrid, 1, 0);


    for (double sflocal : shiftfactorarray)
    {
        shiftfactor = sflocal;
        set_shiftvalue(shiftfactor);

        for (int mcslocal : mcsamplesarray)
        {
            mc_sample = mcslocal;

            for (double cliplocal : CUTCOEFFarray)
            {
                CUTCOEFF = cliplocal;


                // Matrix construction
                Poisson poisson(grid);
                HelmHoltz rhs(grid);
                StencilMC<double (*)(double *)> stencilVarCoeff(grid,&var_coeff,mc_sample);
                LocalStiffnessMatricesDynamicDistribution lhs(grid, stencilVarCoeff,numberLSprocesses);
                lhs.addStencil(poisson);


                // fields for solving
                double eigenvalue_power;
                int pow_interations;
                double cg_interations;
                double time_power;
                double precon = 0;
                VectorSparseG prew_inv(grid);
                prew_inv = 1.0;

                // SOLVING
                Power::power_inverse<LocalStiffnessMatricesDynamicDistribution, HelmHoltz>(eps, prew_inv, eigenvalue_power, pow_interations, cg_interations,
                    m, lhs, rhs, true, precon, time_power);

                // Undoing the shift
                eigenvalue_power /= (rhs_factor * shiftfactor);


                // LOGGING results
                string legend = "level\tDOFS\t\tshift\tclip\tmcsampl\t\teig\t\teig_diff\t\tcg_iter\t\tpow_iter\ttime";
                string logstr = to_string(lvl) + "\t" + to_string(grid.getDOFS()) + "\t\t";
                logstr += to_string(shiftfactor) + "\t" + to_string(CUTCOEFF) + "\t" + to_string(mc_sample) + "\t\t";
                logstr += to_string(eigenvalue_power) +"\t\t" + to_string(eigenvalue_power-eig_exp) + "\t\t";
                logstr += to_string(cg_interations) + "\t\t" + to_string(pow_interations) + "\t" + to_string(time_power);
                mpi_cout(logstr);
            }
        }
    }


#ifdef MY_MPI_ON
    MPI_Finalize();
#endif

    return 0;


}