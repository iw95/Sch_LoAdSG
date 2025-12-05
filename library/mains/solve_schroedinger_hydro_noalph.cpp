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



double border = 8e-12;
//3e-9;
//8e-10;

double eps = 1e-10;

const double CUTCOEFF = 1e25;

    // https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
    double bohr_radius = 5.29177210544e-11;
    double b_factor = sqrt(M_PI * bohr_radius * bohr_radius * bohr_radius);


    // https://physics.nist.gov/cgi-bin/cuu/Value?me
    const double electronmass = 9.1093837139;
    const double orderemass = 1e-31;

    // https://physics.nist.gov/cgi-bin/cuu/Value?e
    const double elementarycharge = 1.602176634;
    const double orderecharge = 1e-19;

    // https://physics.nist.gov/cgi-bin/cuu/Value?h
    const double planckconstant = 6.62607015;
    const double orderplanck = 1e-34;

    // https://physics.nist.gov/cgi-bin/cuu/Value?ep0
    const double vacuumpermitivity = 8.8541878188;
    const double ordereps0 = 1e-12;


    const double val_factor = -2 * border * electronmass * elementarycharge * elementarycharge / 
                        (planckconstant * planckconstant * M_PI * vacuumpermitivity);
    const double order_factor = (orderemass / orderplanck) * (orderecharge / orderplanck) * (orderecharge / ordereps0);

    const double varc_factor = val_factor * order_factor;


    const double real_rhs_factor = 4 * 2*electronmass / (planckconstant * planckconstant) * (border*border*orderemass / (orderplanck*orderplanck));

    // ATTENTION: Changing rhs_factor
    const double rhs_factor = real_rhs_factor; //real_rhs_factor * 1e-10;






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





double var_coeff( double* coordinates)
{
    double r = 0;
    for (int i = 0; i < DimensionSparseGrid; i++) {
        double ci = 2 * coordinates[i] - 1;
        r += ci * ci;
    }

    r = sqrt(r);

        
    double result = CUTCOEFF;
    if (r!=0){
        result = min(varc_factor / r, CUTCOEFF); // < clipping large values of 1/r around the center
    }
    


    return result;
}




inline double unitSQ2atom(double coordinate) {
    return (2 * coordinate - 1) * border;
}


double known_eigenfunction(IndexDimension I) {
    // Calculate distance from zero r
    double r = 0;
    for(int d=0; d<DimensionSparseGrid; d++){
        const double atom_coord = unitSQ2atom(I.coordinate(d));
        r += atom_coord * atom_coord;
    }
    r = sqrt(r);

    double val = exp(- 1 * r / bohr_radius) / b_factor;


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
        cout << "And Coefficient cutoff = " << CUTCOEFF << endl << endl;
        cout << "Factor on rhs of eq = " << rhs_factor << endl << endl;

        cout << "Level\tDOFs\tproblem\t\tmult\t\tpower" << endl;
    }



    int level_start=1;

    AdaptiveSparseGrid grid;
    IndexDimension centerPoint;



    double Linfty_old = 1.0;

    for (int level = level_start; level <10; level++)
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



        for (unsigned long i = 0; i < grid.getMaximalOccupiedSecondTable(); i++) {
                IndexDimension I = grid.getIndexOfTable(i);
                double val_u = known_eigenfunction(I);
                u.setValue(i,val_u);
        }
        calcPrewByNodal(prew_u, u);





        
        Poisson poisson(grid);

        HelmHoltz rhs(grid);

        StencilMC<double (*)(double *)> stencilVarCoeff(grid,&var_coeff,1);
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



        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\tvarc\t\t" + to_string(eigenvalue_mult), false);
        mpi_cout("\t" + to_string(eigenvalue_power));



        string path = "/home/cip/2018/iw95yxig/ciptmp/LoAdSG/library/mains/vector_log/";
        prew_u = prew_u / product(prew_u, prew_u);
        calcNodalByPrew(prew_u, u);
        u.Print_gnu(path + "book_" + to_string(level) + ".dat");
        prew_inv = prew_inv / product(prew_inv, prew_inv);
        calcNodalByPrew(prew_inv, u);
        u.Print_gnu(path + "power_" + to_string(level) + ".dat");






        double p_eig_mult = Power::eigenvalue_exp<Poisson,HelmHoltz>(m, poisson, rhs, prew_u, grid);
        double p_eig_power;
        Power::power_inverse<Poisson, HelmHoltz>(eps, prew_inv, p_eig_power, pow_interations, cg_interations,
            m, poisson, rhs, true, precon, time_power);



        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\thelmh\t\t" + to_string(p_eig_mult), false);
        mpi_cout("\t" + to_string(p_eig_power));



        LocalStiffnessMatricesDynamicDistribution lhs_wo_po(grid, stencilVarCoeff,numberLSprocesses);
        double ov_eig_mult = Power::eigenvalue_exp<LocalStiffnessMatricesDynamicDistribution,HelmHoltz>(m, lhs_wo_po, rhs, prew_u, grid);
        double ov_eig_power;
        Power::power_inverse<LocalStiffnessMatricesDynamicDistribution,HelmHoltz>(eps, prew_inv, p_eig_power, pow_interations, cg_interations,
            m, lhs_wo_po, rhs, true, precon, time_power);


        mpi_cout(to_string(level) + "\t" + to_string(grid.getDOFS()) + "\to/varco\t\t" + to_string(ov_eig_mult), false);
        mpi_cout("\t" + to_string(ov_eig_power));

        mpi_cout("");

    }
#ifdef MY_MPI_ON
    MPI_Finalize();
#endif
    return 0;


}