//
// Created by rumi on 17/12/25.
//

#include "../source/sparseEXPDE.h"
#include "../source/iterator/depthIterator.h"
#include "../source/applications/power_method.h"
//#define testNOW
//#define goolePerf
#include <string>
#include <ctime>
#include <sys/time.h>

const int mc_samples = 100;
const double eps = 1e-30;

int num_tasks = 1;
int numberMVprocesses=1;
int numberLSprocesses;


const int num_samples = 9;


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


double var_coeff0(double* coordinates)
{
    return 1.;
}


double var_coeff1(double* coordinates)
{
    return 6400 * (coordinates[1]-0.5)*(coordinates[1]-0.5);
}


double factor2 = 80.;
double var_coeff2(double* coordinates)
{
    double result = 0;
    result += (coordinates[0]-0.5)*(coordinates[0]-0.5);
    result += (coordinates[1]-0.5)*(coordinates[1]-0.5);
    result *= factor2*factor2*4;;
    return result;
}


double factor3 = 40.;
double var_coeff3(double* coordinates)
{
    double result = 0;
    result += (coordinates[0]-0.5)*(coordinates[0]-0.5);
    result += (coordinates[1]-0.5)*(coordinates[1]-0.5);
    result *= factor3*factor3*4;
    return result;
}


double border4 = 5.;
double var_coeff4(double* coordinates)
{
    double result = 0;
    result += (2*coordinates[0] - 1) * (2*coordinates[0] - 1);
    result += (2*coordinates[1] - 1) * (2*coordinates[1] - 1);
    result *= 16*border4*border4*border4*border4;
    return result;
}


double border5 = 5.;
double var_coeff5(double* coordinates)
{
    double result = 0;
    result += (2*coordinates[0] - 1) * (2*coordinates[0] - 1);
    result += (2*coordinates[1] - 1) * (2*coordinates[1] - 1);
    if (result > 1e6) result = 1e6;
    result = 1/sqrt(result);
    result *= -4*border5;
    return result;
}


double var_coeff6(double* coordinates)
{
    double result = var_coeff5(coordinates);
    result += 4*border5*border5*1.9*1;
    return result;
}


double border7 = 20;
double var_coeff7(double* coordinates)
{
    double result = 0;
    result += (2*coordinates[0] - 1) * (2*coordinates[0] - 1);
    result += (2*coordinates[1] - 1) * (2*coordinates[1] - 1);
    if (result > 1e6) result = 1e6;
    result = 1/sqrt(result);
    result *= -4*border7;
    return result;
}


double var_coeff8(double* coordinates)
{
    double result = var_coeff7(coordinates);
    result += 4*border7*border7*1.9*1;
    return result;
}



double expected_eig[] = {
                        2*M_PI*M_PI+1,
                        M_PI*M_PI+80,
                        factor2*4,
                        factor3*4,
                        16*border4*border4,
                        -4*border5*border5,
                        (-1+1*1.9)*4*border5*border5,
                        -4*border7*border7,
                        (-1+1*1.9)*4*border7*border7
                        };



template<double var_coeff (double*)>
string execute_sample(AdaptiveSparseGrid &grid, MatrixVectorHomogen &m) {

    HelmHoltz rhs(grid);
    Poisson poisson(grid);

    StencilMC<double (*)(double *)> stencilVarCoeff(grid,var_coeff,mc_samples);
    LocalStiffnessMatricesDynamicDistribution lhs(grid, stencilVarCoeff,numberLSprocesses);

    lhs.addStencil(poisson);



    // VectorSparseGrid prew_u(grid);
    //double eigenvalue_mult = Power::eigenvalue_exp<LocalStiffnessMatricesDynamicDistribution,HelmHoltz>(m, lhs, rhs, prew_u, grid);



    double eigenvalue_power;

    int pow_interations;
    double cg_interations;
    double time_power;
    double precon = 0;
    VectorSparseG prew_inv(grid);
    prew_inv = 1.0;

    Power::power_inverse<LocalStiffnessMatricesDynamicDistribution, HelmHoltz>(eps, prew_inv, eigenvalue_power, pow_interations, cg_interations,
                m, lhs, rhs, true, precon, time_power);



    string res = "";
    //mults_str += "\t" + to_string(eigenvalue_mult);
    res += "\t" + to_string(eigenvalue_power);
    return res;
}





int main(int argc, char **argv) {

    int rank = 0;


    #ifdef MY_MPI_ON
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

        numberLSprocesses=num_tasks - numberMVprocesses;
    #endif



    mpi_cout(" Dimension " + to_string( DimensionSparseGrid));
    mpi_cout("regular grid, solve: -lap u + c*u = f ");


    #pragma omp parallel
        {
    #pragma omp single
            {
                int total_threads = omp_get_num_threads();
                mpi_cout("number of mpi_tasks " + to_string(num_tasks));
                mpi_cout("number of omp_threads " + to_string(total_threads));
            }
        }




    if (rank == 0) {
        cout << "use " << numberLSprocesses << " for LocalStiffnessmatrices " << endl;
        cout << "use " << (num_tasks - numberLSprocesses) << " for Matrix Vector Multiplication " << endl << endl;

        cout << "With epsilon = " << eps << endl;
        cout << "Monte Carlo samples = " << mc_samples << endl << endl;

        string legend = "level\tDOFS\tproblem";
        string str_exp_eig = "\texpected";
        for (int pr = 0; pr < num_samples; pr += 1) {
            legend += "\t\t\t" + to_string(pr);
            str_exp_eig += "\t\t" + to_string(expected_eig[pr]);
        }
        cout << legend << endl << endl << str_exp_eig << endl << endl;
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

        string powers_str = "";
   
        powers_str += "\t" + execute_sample<&var_coeff0>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff1>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff2>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff3>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff4>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff5>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff6>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff7>(grid, m);
        powers_str += "\t" + execute_sample<&var_coeff8>(grid, m);


        // mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\t", false);
        // mpi_cout("mult\t", false);
        // mpi_cout(mults_str);

        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\t", false);
        mpi_cout(powers_str);

        mpi_cout("");

    }

#ifdef MY_MPI_ON
    MPI_Finalize();
#endif
    return 0;


}