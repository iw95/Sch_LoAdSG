//
// Created by rumi on 23/09/25.
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

    // https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
    double bohr_radius = 5.29177210544e-11;
    double b_factor = sqrt(M_PI);


    const double varc_factor = -8*border;

    const double rhs_factor = 4*border*border;

    const double eig_exp = -1.*rhs_factor;


double alpha = 1.;


volatile double r_min = 1;
volatile double r_max = 1;




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




string r_minmax_res_log() {
    string str = to_string(r_min) + "\t" + to_string(r_max);
    r_min = 1;
    r_max = 1;
    return str;
}





double var_coeff( double* coordinates)
{
    double r = 0;
    for (int i = 0; i < DimensionSparseGrid; i++) {
        double ci = 2. * coordinates[i] - 1.; // < map from [0,1] to [-s,s]
        r += ci * ci;
    }

    r = sqrt(r);

        
    double result = CUTCOEFF;
    if (r!=0){
        result = min(1. / r, CUTCOEFF); // < clipping large values of 1/r around the center
    }

    result = varc_factor * result;
    result -= 1.9*eig_exp; // + 1.9*1*4*s^2 (eig_exp==1)


    double over_r = 1./r;
    if (over_r < r_min) {
        r_min = over_r;
    }
    if (over_r > r_max) r_max = over_r;

    //cout << over_r << "\t";



    return alpha*result;
}




inline double unitSQ2atom(double coordinate) {
    return (2. * coordinate - 1.) * border;
}


double known_eigenfunction(IndexDimension I) {
    // Calculate distance from zero r
    double r = 0.;
    for(int d=0; d<DimensionSparseGrid; d++){
        const double atom_coord = unitSQ2atom(I.coordinate(d));
        r += atom_coord * atom_coord;
    }
    r = sqrt(r);

    double val = exp(- 1. * r) / b_factor;


    return val;
}


double laplace_sin(IndexDimension I) {
    double val = 1.;
    for (int d=0; d<DimensionSparseGrid; d++) {
        val *= sin(M_PI * I.coordinate(d));
    }
    return val;
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
    mpi_cout("regular grid, solve: -lap u + (c-1.9lambda)*u =  -0.9lamda");


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
        cout << "Computing the first eigenvalue of the Schroedinger equation for the Spinless Hydrogen atom." << endl;
        cout << "The expected eigenvalue is " << -13.6 << endl << endl;
        cout << "Dimension " << DimensionSparseGrid << endl;
        cout << "Computing on domain [-" << border << ", " << border << "]" << endl;
        cout << "With epsilon = " << eps << endl;
        cout << "And Coefficient cutoff = " << CUTCOEFF << endl;
        cout << "Monte Carlo samples = " << mc_samples << endl << endl;
        //mpi_cout("level\tDOFS\tSummed\t\tPoisson\t\tVarC\t\t\tDiffADD");

        cout << "Factor k^2 = " << varc_factor << endl << endl;

        string legend = "level\tDOFS\t\t\tHelmholz\talph=";
        for (double alpha_local = 0.; alpha_local <= 1; alpha_local += 0.1) {
            legend += "\t" + to_string(alpha_local);
        }
        legend += "\t\tcg_iter";
        cout << legend << endl << endl;
        //cout <<"level\tDOFS\tByMult\t\tPower" << endl;
    }







    int level_start=1;

    AdaptiveSparseGrid grid;
    IndexDimension centerPoint;


    string minmax_str = "level\t\tmin\tmax\n";


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


        VectorSparseG u(grid);
        VectorSparseG prew_u(grid);


        double val_u = 0.0;


        for (unsigned long i = 0; i < grid.getMaximalOccupiedSecondTable(); i++) {
                IndexDimension I = grid.getIndexOfTable(i);
                val_u = known_eigenfunction(I);
                u.setValue(i,val_u);
        }
        calcPrewByNodal(prew_u, u);






        string mults_str = "";
        string powers_str = "";
        



        VectorSparseG u_noalph(grid);
        VectorSparseG prew_u_noalph(grid);
                for (unsigned long i = 0; i < grid.getMaximalOccupiedSecondTable(); i++) {
                    IndexDimension I = grid.getIndexOfTable(i);
                    u_noalph.setValue(i,laplace_sin(I));
            }
        calcPrewByNodal(prew_u_noalph, u_noalph);
        HelmHoltz rhs_noalph(grid);
        Poisson lhs_noalph(grid);
        double ev_noalph_mult = Power::eigenvalue_exp<Poisson,HelmHoltz>(m, lhs_noalph, rhs_noalph, prew_u_noalph, grid);
        double ev_noalph_power;
            int pow_interations;
            double cg_interations;
            double time_power;
            double precon = 0;
            VectorSparseG prew_inv(grid);
            prew_inv = 1.0;
        Power::power_inverse<Poisson, HelmHoltz>(eps, prew_inv, ev_noalph_power, pow_interations, cg_interations,
                m, lhs_noalph, rhs_noalph, true, precon, time_power);

        mults_str += "\t" + to_string(ev_noalph_mult) + "\t";
        powers_str += "\t" + to_string(ev_noalph_power) + "\t";




        
        Poisson poisson(grid);
            

        double alphasteps = 10;
        for (int alpha_local = 0; alpha_local <= alphasteps; alpha_local += 1) {

            alpha = alpha_local / alphasteps;


            HelmHoltz rhs(grid);

            StencilMC<double (*)(double *)> stencilVarCoeff(grid,&var_coeff,mc_samples);
            LocalStiffnessMatricesDynamicDistribution lhs(grid, stencilVarCoeff,numberLSprocesses);
            lhs.addStencil(poisson);



            double eigenvalue_mult = Power::eigenvalue_exp<LocalStiffnessMatricesDynamicDistribution,HelmHoltz>(m, lhs, rhs, prew_u, grid);



            double eigenvalue_power;

            int pow_interations;
            double cg_interations;
            double time_power;
            double precon = 0;
            VectorSparseG prew_inv(grid);
            prew_inv = 1.0;

            Power::power_inverse<LocalStiffnessMatricesDynamicDistribution, HelmHoltz>(eps, prew_inv, eigenvalue_power, pow_interations, cg_interations,
                m, lhs, rhs, true, precon, time_power);


            mults_str += "\t" + to_string(eigenvalue_mult/rhs_factor);
            powers_str += "\t" + to_string(eigenvalue_power/rhs_factor);

            if (alpha_local == alphasteps) {
                powers_str += "\t\t" + to_string(cg_interations);
            }
        
        }


        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\t", false);
        mpi_cout("mult\t", false);
        mpi_cout(mults_str);

        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\t", false);
        mpi_cout("power\t", false);
        mpi_cout(powers_str);

        mpi_cout("");


        minmax_str += to_string(level) + "\t\t" + r_minmax_res_log() + "\n";


    }

    #ifdef MY_MPI_ON
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank==1)
            cout << minmax_str;
    #endif

#ifdef MY_MPI_ON
    MPI_Finalize();
#endif
    return 0;


}