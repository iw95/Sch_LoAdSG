//
// Created by to35jepo on 3/7/23.
//

#ifndef SGRUN_STENCIL_H
#define SGRUN_STENCIL_H

#include "../indices/index.h"




#include "InterfaceIntegration/interfaceMatrices.h"
#include "InterfaceIntegration/constantIntegrators.h"
#include "InterfaceIntegration/interatorBasisFunction.h"
#include "InterfaceIntegration/HelmholtzIntegrator/helmIntegrator.h"

#include "../cells/celldimension.h"
#include "../cells/iterator_neu.h"
#include "../applications/norms.h"
#include "../iterator/RectangularIterator.h"
#include "PoissonStencil.h"


#include <omp.h>

#include <sstream>








template <class Stencilclass>
double getLocalStencilBoundary(IndexDimension Index, Depth &T, VectorSparseG &u, Stencilclass stencilclass) {


    stencilclass.initialize(T);

    double val = 0.0;
    IndexDimension NextTest;

    for (MultiDimCompass mc; mc.goon(); ++mc) {
        NextTest = Index.nextThree_boundary(&mc, T.returnTiefen());

        if(mc.goon()) {
            double value = 0.0;
            value = stencilclass.returnValue(Index, mc);
            val = val + value * u.getValue(NextTest);
        }
    }
    return val;

}


class CellIterator{
public:
    CellIterator(MultiDimCompass mc, Depth T, IndexDimension center){
        int val=1;

        // Berechne Anzahl der Zellen, die MC berühren
        for(int d=0; d<DimensionSparseGrid; d++){
            if(mc.getRichtung(d)==Mitte&&center.isNotAtBoundary(d))val=2*val;
            if(mc.getRichtung(d)==Links&&center.isAtLeftBoundary(d)) val=0.0;
            if(mc.getRichtung(d)==Rechts&&center.isAtRightBoundary(d)) val=0.0;
        }
        maxShift = val;

        Indices = new IndexDimension[maxShift];
        recursive(center,0,mc,T);
    }
    ~CellIterator(){
        delete[] Indices;
    }

    void PrintIndices(){
        for(int j=0; j<maxShift;j++){
            Indices[j].PrintCoord();
        }
    }

    int maxShift;

    bool goon(){
        return (shiftNumber<maxShift);
    }

    void operator++(){
        shiftNumber++;
    }

    IndexDimension getCell(){
        return Indices[shiftNumber];
    }

    void start(){
        shiftNumber=0;
    }

private:

    IndexDimension* Indices;
    int k = 0;
    int shiftNumber=0;

    void recursive(IndexDimension I, int d, MultiDimCompass mc, Depth T){
        if(d == DimensionSparseGrid) {
            Indices[k]=I;
            k++;
        }else {
            int D = d+1;
            if (mc.getRichtung(d) == Mitte) {
                // rechts
                recursive(I, D, mc,T);
                //links
                if(!I.isAtLeftBoundary(d)){
                    I = I.nextLeft(d, T.at(d));
                    recursive(I, D, mc, T);
                }
            }
            if (mc.getRichtung(d) == Links && !I.isAtLeftBoundary(d)) {
                I = I.nextLeft(d, T.at(d));
                recursive(I, D, mc,T);
            }
            if (mc.getRichtung(d) == Rechts &&!I.isAtRightBoundary(d)) {
                recursive(I, D, mc,T);
            }
        }
    }
};




class  StencilTemplate{

    template<class StencilA, class StencilB>
    friend class ADD;


    public:
    // ghost functions and members
    virtual void resetActiveWorkers(){};
    virtual void receiveApplySendOnActiveWorkers(){};
    virtual int getNumberProcesses(){return 0;};
    virtual int size(){return 0;};
    std::vector<int> active_worker;
    virtual void applyLocalStiffnessMatricesDepth(VectorSparseG& u,VectorSparseG& z,Depth& depth){};
    virtual double applyLocalStiffnessMatricesFixedDepthIndex_onNode(VectorSparseG& input, VectorSparseG& output, Depth& D, IndexDimension& Index){return 0.0;};
    virtual int getNodeForDepth(Depth& T){return 0;};
    void applyLocalStiffnessMatricesOnIndices_onNode(VectorSparseG& input, VectorSparseG& output, Depth& D, std::vector<IndexDimension>& Indices){};


    StencilTemplate(AdaptiveSparseGrid& grid_): grid(grid_){};

    virtual inline void initialize(Depth &T_) {
        T = T_;

        for (int d = 0; d < DimensionSparseGrid; d++) {
            int exp = T_.at(d);
            if (exp == 0) meshwidth[d] = 1.0;
            else {

                int value2 = 1 << exp;
                exponent[d] = double(value2);
                meshwidth[d] = 1.0 / double(value2);
            }
        }

    };

    inline void copyInitialization(StencilTemplate& stencil){
        T = stencil.T;

        for (int d = 0; d < DimensionSparseGrid; d++) {
           meshwidth[d]=stencil.meshwidth[d];
           exponent[d]=stencil.exponent[d];

        }
    }



    virtual inline void applyStencilOnCell(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){

        for(CellIndexIterator outerIter(&cell); outerIter.goon(); ++outerIter) {


            IndexDimension p = outerIter.getIndex();

            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter.getCellIndexDirection();


            if (occP) {

                for (CellIndexIterator innerIter(outerIter); innerIter.goon(); ++innerIter) {


                    IndexDimension q = innerIter.getIndex();
                    if (q.isNotAtBoundary()) {

                        CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                        unsigned long kq;

                        bool occQ = output.getSparseGrid()->occupied(kq, q);
                        if (occQ) {

                            double val = integration(cell, dirP, dirQ);
                            if (p == q) {
                                val = val * input.getValue(kp);


                                output.addToValue(kq, val);
                            } else {
                                double val_p = val * input.getValue(kp);

                                output.addToValue(kq, val_p);

                                double val_q = val * input.getValue(kq);

                                output.addToValue(kp, val_q);
                            }
                        }

                    }

                }

            }
        }









    }

    virtual inline void applyStencilOnCell_BinaryIterator(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){



        int iterend = PowerOfTwo<DimensionSparseGrid>::value;

        CellIndexIterator outerIter(&cell);
        for(int iteri=0; iteri<iterend ; iteri++) {
            IndexDimension p = outerIter.getIndexByBinary(iteri);
            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter.getCellIndexDirection();
            if (occP) {
                for (int iterj = iteri; iterj < iterend; iterj++) {
                    IndexDimension q = outerIter.getIndexByBinary(iterj);
                    if (q.isNotAtBoundary()) {
                        CellIndexDirection dirQ = outerIter.getCellIndexDirection();
                        unsigned long kq;

                        bool occQ = output.getSparseGrid()->occupied(kq, q);
                        if (occQ) {

                            double val = integration(cell, dirP, dirQ);
                            if (p == q) {
                                val = val * input.getValue(kp);

                                //#pragma omp critical
                                output.addToValue(kq, val);
                            } else {
                                double val_p = val * input.getValue(kp);
                                //#pragma omp critical
                                output.addToValue(kq, val_p);

                                double val_q = val * input.getValue(kq);
                                //#pragma omp critical
                                output.addToValue(kp, val_q);
                            }
                        }

                    }
                }
            }
        }

    }

    virtual inline void applyStencilOnCell_nosymmetry(CellDimension& cell,VectorSparseG& input, VectorSparseG& output) {


        for (CellIndexIterator outerIter(&cell); outerIter.goon(); ++outerIter) {


            IndexDimension p = outerIter.getIndex();

            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter.getCellIndexDirection();


            if (occP) {



                 for (CellIndexIterator innerIter(&cell); innerIter.goon(); ++innerIter) {

                      IndexDimension q = innerIter.getIndex();
                      if(q.isNotAtBoundary()){
                          CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                          unsigned long kq;

                          bool occQ= output.getSparseGrid()->occupied(kq,q);
                          if(occQ){


                              double val = integration(cell, dirP, dirQ);
                              val =val*input.getValue(kp);
                              output.addToValue(kq, val);


                          }
                      }
                  }           }
        }
    }

    virtual inline void applyStencilOnCellMPI(CellDimension& cell,VectorSparseG& input, VectorSparseG& output) {


    int iterend = POW2(DimensionSparseGrid);


        {
            CellIndexIterator outerIter(&cell);

            for (int iteri = 0; iteri < iterend; iteri++) {


                IndexDimension p = outerIter.getIndex();

                unsigned long kp;
                bool occP = output.getSparseGrid()->occupied(kp, p);
                CellIndexDirection dirP = outerIter.getCellIndexDirection();


                if (occP) {
                    for (CellIndexIterator innerIter(outerIter); innerIter.goon(); ++innerIter) {
                        IndexDimension q = innerIter.getIndex();
                        if (q.isNotAtBoundary()) {
                            CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                            unsigned long kq;

                            bool occQ = output.getSparseGrid()->occupied(kq, q);
                            if(occQ){
                                double val = integration(cell, dirP, dirQ);
                                if(p==q){
                                    val =val*input.getValue(kp);

                                    output.addToValue(kq, val);
                                }else{
                                    double val_p =val*input.getValue(kp);

                                    output.addToValue(kq, val_p);

                                    double val_q =val*input.getValue(kq);

                                    output.addToValue(kp, val_q);
                                }
                            }

                        }

                    }


                }
                ++outerIter;
            }
        }
    }


    virtual inline void applyStencilOnCell_MPI_OMP(CellDimension& cell,VectorSparseG& input, VectorSparseG& output) {


        int iterend = POW2(DimensionSparseGrid);
        CellIndexIterator outerIter(&cell);

        int entries = iterend*(iterend+1)/2;
        CellIndexDirection pCellIndexDirection[2][entries];
        bool calcIntgral[entries];
        bool diagonalEntry[entries];
        unsigned long occupied[2][entries];
        double values[2][entries];
        int j=0;


        for (CellIndexIterator outerIter2(&cell); outerIter2.goon(); ++outerIter2){

            IndexDimension p = outerIter2.getIndex();
            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter2.getCellIndexDirection();


                for (CellIndexIterator innerIter(outerIter2); innerIter.goon(); ++innerIter) {
                    IndexDimension q = innerIter.getIndex();

                    if (q.isNotAtBoundary()) {

                        CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                        unsigned long kq;

                        bool occQ = output.getSparseGrid()->occupied(kq, q);
                        if (occQ && occP) {
                            calcIntgral[j] = true;
                            pCellIndexDirection[0][j] = dirP;
                            pCellIndexDirection[1][j] = dirQ;
                            occupied[0][j]=kp;
                            occupied[1][j]=kq;
                            if (p == q) {
                                diagonalEntry[j] = true;


                            } else {
                                diagonalEntry[j] = false;
                            }
                        } else {
                            calcIntgral[j] = false;
                        }

                    } else {
                        calcIntgral[j] = false;
                    }
                    j++;
                }
        }


#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < entries; i++) {
            if (calcIntgral[i]) {
                double val = integration(cell, pCellIndexDirection[0][i], pCellIndexDirection[1][i]);
                unsigned long kp = occupied[0][i];
                unsigned long kq = occupied[1][i];
                if (diagonalEntry[i]) {
                    val = val * input.getValue(kp);
                    values[0][i] = val;

                } else {
                    double val_p = val * input.getValue(kp);
                    values[1][i] = val_p;


                    double val_q = val * input.getValue(kq);

                    values[0][i] = val_q;

                }

            }
        }





        for (int i = 0; i < entries; i++) {
            if (calcIntgral[i]) {
                unsigned long kp = occupied[0][i];
                unsigned long kq = occupied[1][i];
                if (diagonalEntry[i]) {

                    double val =values[0][i];

                    output.addToValue(kq, val);
                } else {
                    //double val_p = val * input.getValue(kp);
                    double val_p =values[1][i];


                    output.addToValue(kq, val_p);

                    double val_q=  values[0][i];
                    output.addToValue(kp, val_q);
                }

            }
        }

    }


    virtual inline void applyStencilOnCell_MPI_OMP_nosymmetry(CellDimension& cell,VectorSparseG& input, VectorSparseG& output) {



        int iterend = POW2(DimensionSparseGrid);

        CellIndexIterator outerIter(&cell);

        int entries = iterend*iterend;
        CellIndexDirection pCellIndexDirection[2][entries];
        bool calcIntgral[entries];

        unsigned long occupied[2][entries];
        double values[entries];
        int j=0;


        for (CellIndexIterator outerIter2(&cell); outerIter2.goon(); ++outerIter2){

            IndexDimension p = outerIter2.getIndex();
            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter2.getCellIndexDirection();


          /*  for (CellIndexIterator innerIter(&cell); innerIter.goon(); ++innerIter) {
                IndexDimension q = innerIter.getIndex();
                if (q.isNotAtBoundary()) {
                    CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                    unsigned long kq;

                    bool occQ = output.getSparseGrid()->occupied(kq, q);
                    if (occQ) {


                        double val = integration(cell, dirP, dirQ);

                        val = val * input.getValue(kp);
                        output.addToValue(kp, val);


                    }

                }
            }*/
                for (CellIndexIterator innerIter(&cell); innerIter.goon(); ++innerIter) {
                    IndexDimension q = innerIter.getIndex();

                    if (q.isNotAtBoundary() && p.isNotAtBoundary()){
                        CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                        unsigned long kq;

                        bool occQ = output.getSparseGrid()->occupied(kq, q);
                        if (occQ && occP) {
                            calcIntgral[j] = true;
                            pCellIndexDirection[0][j] = dirP;
                            pCellIndexDirection[1][j] = dirQ;
                            occupied[0][j]=kp;
                            occupied[1][j]=kq;
                        } else {
                            calcIntgral[j] = false;
                        }

                    } else {
                        calcIntgral[j] = false;
                    }
                    j++;
                }

        }


#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < entries; i++) {
            if (calcIntgral[i]) {
                double val = integration(cell, pCellIndexDirection[0][i], pCellIndexDirection[1][i]);
                unsigned long kp = occupied[0][i];
                //unsigned long kq = occupied[1][i];

                double val_p = val * input.getValue(kp);

                values[i] = val_p;
            }
        }


        for (int i = 0; i < entries; i++) {
            if (calcIntgral[i]) {
               //unsigned long kp = occupied[0][i];
                unsigned long kq = occupied[1][i];
                double val_p =values[i];
                output.addToValue(kq, val_p);
            }
        }

/*
            //#pragma omp parallel for schedule(runtime)
            for (int iteri=0; iteri < iterend; iteri++) {
                int thread_num = omp_get_thread_num();


                IndexDimension p = outerIter.getIndex(iteri);



                unsigned long kp;
                bool occP = output.getSparseGrid()->occupied(kp, p);
                CellIndexDirection dirP = outerIter.getCellIndexDirection(iteri);



                if (occP) {
                    for (CellIndexIterator innerIter(outerIter,iteri); innerIter.goon(); ++innerIter) {
                        IndexDimension q = innerIter.getIndex();


                        if (q.isNotAtBoundary()) {

                            CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                            unsigned long kq;

                            bool occQ = output.getSparseGrid()->occupied(kq, q);
                            if(occQ){
                                double val = integration(cell, dirP, dirQ);
                                if(p==q){
                                    val =val*input.getValue(kp);


                                    output.addToValue(kq, val);
                                }else{
                                    double val_p =val*input.getValue(kp);


                                    output.addToValue(kq, val_p);

                                    double val_q =val*input.getValue(kq);


                                    output.addToValue(kp, val_q);
                                }
                            }

                        }

                    }


                }

                ++outerIter;

            }
            output_test=output_test-output;
            if(L_infty(output_test)>1e-20){

                exit(0);}*/


    }


    virtual inline double returnValue(const IndexDimension &Index, const MultiDimCompass &mc){


        double val = 0.0;



        CellIterator cells(mc,T,Index);


        for(cells.start(); cells.goon();++cells){
            IndexDimension cell = cells.getCell();
            if(cellInGrid(cell)) {


                val += localValue(Index, cell, mc);




            }

        }





        return val;



    };



    bool cellInGrid(IndexDimension cell) const{
        IndexDimension Imin = cell;
        IndexDimension Imax = cell;

        for(int d=0; d<DimensionSparseGrid; d++){
            Imax = Imax.nextRight(d,T.at(d));
        }

        RectangularIterator iterator(Imin,Imax,T);
        for(;iterator.goon();++iterator){
            IndexDimension I = iterator.getIndex();
            unsigned long k;
            if(I.isNotAtBoundary() && !(grid.occupied(k,I))) return false;
        }

        return  true;
    }
    /**
   *
   *    /\  /\  /\
   *   /  \/  \/  \
   *  /   /\  / \  \
   * /  /   \/   \  \
   *  l   cL  cR   r
   *
   *
   *
   * @param center
   * @param cell
   * @param mc
   * @return
   */
    virtual inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{
        //cout << "value " << endl;
        return 0.0;
    };

    virtual inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){

        cout << "not implemented yet! " << endl;
        exit(1);
        return 0.0;
    }
    Depth T;

    virtual TypeMatrixVectorMultiplication getTypeMatrixVectorMultiplication(){return typeMatrixVectorMultiplication;};


protected:




    // integrate (-x+p)*(x-(p-h))
    inline double integral_left(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p=center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        return val2-val1;
    }

    // integrate (-x+(p+h))*(x-p)
    inline double integral_right(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p=center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        return val2-val1;
        //return -(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
    }


    inline double integral_CenterR(const IndexDimension& center,const IndexDimension& cell, int d) const{


        double p =center.coordinate(d);
        double h =meshwidth[d];
        double x = cell.coordinate(d);



        double val1= (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;


        x = x+h;
        double val2=  (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;

        return val2-val1;

    }



    // integrate (x-(p-h))²
    inline double integral_CenterL(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p = center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        x = x+h;
        double val2=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        return val2-val1;

    }


    virtual inline double integral_RR(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;

        //integrate (x-b)² from a to b
        double val1=(1.0/3.0)*h*h*h;

        return val1;
    }


    virtual inline double integral_LL(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;


        //integrate (a-x)² from a to b
        double val1=(1.0/3.0)*h*h*h;

        return val1;
    }

    virtual inline double integral_LR(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;

        double val1=(1.0/6.0)*h*h*h;
        return val1;
    }




    double meshwidth[DimensionSparseGrid];
    double exponent[DimensionSparseGrid];



    AdaptiveSparseGrid& grid;

    TypeMatrixVectorMultiplication typeMatrixVectorMultiplication = StencilOnTheFly;


};



class HelmHoltz: public StencilTemplate{
public:
    HelmHoltz(AdaptiveSparseGrid& grid_): StencilTemplate(grid_){};


    inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {
            if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d))
                value *= integral_CenterR(center,cell, d);
            if (mc.getRichtung(d) == Mitte && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_CenterL(center,cell, d);
            if (mc.getRichtung(d) == Links && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_left(center,cell, d);
            if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d))
                value *= integral_right(center,cell, d);

            value = exponent[d]*exponent[d]*value;
        }




        return value;}

    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){
        double value = 1.0;

        for(int d=0; d<DimensionSparseGrid; d++) {


                if (dirP.getDirection(d)==dirQ.getDirection(d)){
                    value *= integral_RR(cell, d);
                    value = exponent[d] * exponent[d] * value;

                }
                else {
                    value *= integral_LR(cell,d);
                    value = exponent[d] * exponent[d] * value;
                }

        }
        return value;

    }

protected:

    // integrate (-x+p)*(x-(p-h))
    inline double integral_left(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        return val2-val1;
    }

    // integrate (-x+(p+h))*(x-p)
    inline double integral_right(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        return val2-val1;
        //return -(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
    }

    inline double integral_CenterR(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);



        double val1= (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;


        x = x+h;
        double val2=  (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;

        return val2-val1;

    }

    // integrate (x-(p-h))²
    inline double integral_CenterL(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p = center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        x = x+h;
        double val2=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        return val2-val1;

    }




};


class ScaledHelmHoltz: public HelmHoltz{

    const double factor;

public:
    ScaledHelmHoltz(AdaptiveSparseGrid& grid_, double factor_): HelmHoltz(grid_), factor(factor_){};


    inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{
        return factor * HelmHoltz::localValue(center, cell, mc);
    }

    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){
        return factor * HelmHoltz::integration(cell, dirP, dirQ);
    }
};



template <typename F>
class StencilInterface: public StencilTemplate {
public:

    StencilInterface(AdaptiveSparseGrid& grid_, const F &varCoeff) : StencilTemplate(grid_), localStiffnessHelm(
            varCoeff, 1e-8, 0, 8, 1.0e8) {
    };

    inline double
    localValue(const IndexDimension &center, const IndexDimension &cell, const MultiDimCompass &mc) const {

        BasisFunctionType u[DimensionSparseGrid];
        BasisFunctionType v[DimensionSparseGrid];
        double p_left[DimensionSparseGrid];
        double p_right[DimensionSparseGrid];

        //IntegratorHelm<double (*) (const std::array<double, DimensionSparseGrid>&), DimensionSparseGrid> localStiffnessHelm(&var_coeff, 1e-10, 0, 10, 1.0e9 );

        double value = 0.0;
        bool integrate = true;
        for (int d = 0; d < DimensionSparseGrid; d++) {
            p_left[d] = cell.coordinate(d);
            p_right[d] = cell.coordinate(d) + meshwidth[d];

            if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d)) {
                u[d] = leftBasis;
                v[d] = leftBasis;
            } else if (mc.getRichtung(d) == Mitte && cell.getIndex(d) != center.getIndex(d)) {

                u[d] = rightBasis;
                v[d] = rightBasis;
            } else if (mc.getRichtung(d) == Links && cell.getIndex(d) != center.getIndex(d)) {
                u[d] = leftBasis;
                v[d] = rightBasis;
            } else if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d)) {
                v[d] = leftBasis;
                u[d] = rightBasis;
            } else {
                integrate = false;
                break;
            }


        }

        if (integrate) {

            value = localStiffnessHelm.stencil_integration(p_left, p_right, u, v);


        }

        stencil_count++;
        return value;
    };

    static int stencil_count;

    inline double integration(CellDimension &cell, CellIndexDirection &dirP, CellIndexDirection &dirQ) {

        BasisFunctionType u[DimensionSparseGrid];
        BasisFunctionType v[DimensionSparseGrid];
        double p_left[DimensionSparseGrid];
        double p_right[DimensionSparseGrid];

        double value = 0.0;

        for (int d = 0; d < DimensionSparseGrid; d++) {
            p_left[d] = cell.getLeftBoundary(d);
            p_right[d] = cell.getRightBoundary(d);


            if (dirP.getDirection(d) == Left) {
                u[d] = leftBasis;
            } else {
                u[d] = rightBasis;
            }

            if (dirQ.getDirection(d) == Left) {
                v[d] = leftBasis;
            } else {
                v[d] = rightBasis;
            }


        }
        value = localStiffnessHelm.stencil_integration(p_left, p_right, u, v);

        return value;
    }
protected:

    IntegratorHelm<const F, DimensionSparseGrid> localStiffnessHelm;


/*    inline double integration(CellDimension &cell, CellIndexDirection &dirP, CellIndexDirection &dirQ) {

        BasisFunctionType u[DimensionSparseGrid];
        BasisFunctionType v[DimensionSparseGrid];
        double p_left[DimensionSparseGrid];
        double p_right[DimensionSparseGrid];

        double value = 0.0;

        for (int d = 0; d < DimensionSparseGrid; d++) {
            p_left[d] = cell.getLeftBoundary(d);
            p_right[d] = cell.getRightBoundary(d);


            if (dirP.getDirection(d) == Left) {
                u[d] = leftBasis;
            } else {
                u[d] = rightBasis;
            }

            if (dirQ.getDirection(d) == Left) {
                v[d] = leftBasis;
            } else {
                v[d] = rightBasis;
            }


        }
        value = localStiffnessHelm.stencil_integration(p_left, p_right, u, v);
        stencil_count++;
        return value;
    }*/
};

template <class F>
int StencilInterface<F>::stencil_count=0;



/**
 *  Analytical Stencil of \int (1-x²)*v(x) on a cell
 */
class VariableCoefficientMass: public StencilTemplate{

public:
    VariableCoefficientMass(AdaptiveSparseGrid& grid_): StencilTemplate(grid_){};




        inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {
            if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d))
                value *= integral_CenterR(center,cell, d);
            if (mc.getRichtung(d) == Mitte && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_CenterL(center,cell, d);
            if (mc.getRichtung(d) == Links && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_left(center,cell, d);
            if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d))
                value *= integral_right(center,cell, d);

            value = exponent[d]*exponent[d]*value;
        }


        return value;};

    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){
        double value = 1.0;

        for(int d=0; d<DimensionSparseGrid; d++) {


            if (dirP.getDirection(d)==dirQ.getDirection(d)){
                if(dirP.getDirection(d)==Right) {
                    value *= integral_LL(cell, d);


                    value = exponent[d] * exponent[d] * value;
                }
                if(dirP.getDirection(d)==Left) {
                    value *= integral_RR(cell, d);

                    value = exponent[d] * exponent[d] * value;
                }
            }
            else {
                value *= integral_LR(cell,d);

                value = exponent[d] * exponent[d] * value;
            }



        }
        return value;

    }

 protected:


    // integrate (-x+p)*(x-(p-h))*(1-x²)
    inline double integral_left(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];

        double x = cell.coordinate(d);
        double val1=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        double val12=-(1.0/5.0)*x*x*x*x*x+(1.0/4.0)*p*x*x*x*x+(1.0/4.0)*(p-h)*x*x*x*x-(1.0/3.0)*p*(p-h)*x*x*x;

        val1 = val1 - val12;
        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        double val22=-(1.0/5.0)*x*x*x*x*x+(1.0/4.0)*p*x*x*x*x+(1.0/4.0)*(p-h)*x*x*x*x-(1.0/3.0)*p*(p-h)*x*x*x;

        val2= val2 - val22;

        return val2-val1;
    }

    // integrate (x-(p-h))²
    inline double integral_CenterL(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p-h)*x*x*x*x+(1.0/3.0)*(p-h)*(p-h)*x*x*x;
        val1 = val1 - val12;


        x = x+h;
        double val2=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p-h)*x*x*x*x+(1.0/3.0)*(p-h)*(p-h)*x*x*x;
        val2= val2 - val22;

        return val2-val1;

    }

    // integrate (-x+(p+h))²*(1-x²)
    inline double integral_CenterR(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);


        double val1= (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*(p+h)*(p+h)*x*x*x;
        val1 = val1 - val12;

        x = x+h;
        double val2=  (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*(p+h)*(p+h)*x*x*x;
        val2= val2 - val22;
        return val2-val1;

    }


    /**
     *   integrate (-x+(p+h))*(x-p)*(1-x²)
     *   @param center = p
     */
     inline double integral_right(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(1.0/4.0)*p*x*x*x*x-(1.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*p*(p+h)*x*x*x;
        val1 = val1 + val12;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(1.0/4.0)*p*x*x*x*x-(1.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*p*(p+h)*x*x*x;
        val2= val2 + val22;
        return val2-val1;

    }


    inline double integral_RR(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;

        //integrate (x-b)²(1-x²) from a to b
        double val1=(-1.0*h*h*h*(-10.0 + 6.0*a*a + 3.0*a*b + b*b))/30.0;
        return val1;

    }

    inline double integral_LL(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;

        //integrate (a-x)²(1-x²) from a to b
        double val1=(-1.0*h*h*h*(-10.0 + 6.0*b*b + 3.0*a*b + a*a))/30.0;
        return val1;

    }
    inline double integral_LR(CellDimension& cell,int d){
        double b = cell.getRightBoundary(d);
        double a = cell.getLeftBoundary(d);
        double h= b-a;

        //integrate (x-b)(a-x)(1-x²) from a to b
        double val1=(-1.0*h*h*h*(-10.0 + 3.0*a*a + 4.0*a*b + 3.0*b*b))/60.0;

        return val1;

    }


};




/**
 *  Analytical Stencil of \int c*v(x) on a cell
 */
class ConstantCoefficientMass: public StencilTemplate{
public:
    ConstantCoefficientMass(AdaptiveSparseGrid& grid_): StencilTemplate(grid_){};

    double c=1.0;

    inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {
            if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d))
                value *= integral_CenterR(center,cell, d);
            if (mc.getRichtung(d) == Mitte && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_CenterL(center,cell, d);
            if (mc.getRichtung(d) == Links && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_left(center,cell, d);
            if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d))
                value *= integral_right(center,cell, d);

            value = exponent[d]*exponent[d]*value;
        }




        return value;}

    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){
        double value = 1.0;

        for(int d=0; d<DimensionSparseGrid; d++) {


            if (dirP.getDirection(d)==dirQ.getDirection(d)){
                value *= integral_RR(cell, d);
                value = exponent[d] * exponent[d] * value;

            }
            else {
                value *= integral_LR(cell,d);
                value = exponent[d] * exponent[d] * value;
            }

        }
        return c*value;

    }

protected:

    // integrate (-x+p)*(x-(p-h))
    inline double integral_left(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        return val2-val1;
    }

    // integrate (-x+(p+h))*(x-p)
    inline double integral_right(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;

        return val2-val1;
        //return -(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
    }

    inline double integral_CenterR(const IndexDimension& center,const IndexDimension& cell, int d) const{

        double p =center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);



        double val1= (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;


        x = x+h;
        double val2=  (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;

        return val2-val1;

    }

    // integrate (x-(p-h))²
    inline double integral_CenterL(const IndexDimension& center,const IndexDimension& cell, int d) const{
        double p = center.coordinate(d);
        double h=meshwidth[d];
        double x = cell.coordinate(d);
        double val1=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        x = x+h;
        double val2=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;


        return val2-val1;

    }




};

class StencilVariableCoeff{
public:
    inline void initialize(Depth &T_) {

        T = T_;
        poisson.initialize(T);

        for (int d = 0; d < DimensionSparseGrid; d++) {
            int exp = T_.at(d);
            if (exp == 0) meshwidth[d] = 1.0;
            else {

                int value2 = 1 << exp;
                exponent[d] = value2;
                meshwidth[d] = 1 / double(value2);

            }
        }

    };



    inline double returnValue(const IndexDimension &Index, const MultiDimCompass &mc){
        double val = 0.0;


        for(int d=0; d<DimensionSparseGrid; d++)
            coord[d]=Index.coordinate(d);

        CellIterator cells(mc,T,Index);


       for(cells.start(); cells.goon();++cells){
            IndexDimension cell = cells.getCell();
            val+= localValue(Index,cell,mc);

        }




        val += poisson.returnValue(Index,mc);
        return val;



    };
    // integrate (-x+p)*(x-(p-h))*(1-x²)
    double integral_left(IndexDimension& cell, int d) const{

        double p = coord[d];
        double h=meshwidth[d];

        double x = cell.coordinate(d);
        double val1=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        double val12=-(1.0/5.0)*x*x*x*x*x+(1.0/4.0)*p*x*x*x*x+(1.0/4.0)*(p-h)*x*x*x*x-(1.0/3.0)*p*(p-h)*x*x*x;

        val1 = val1 - val12;
        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+0.5*(p-h)*x*x+0.5*p*x*x-p*(p-h)*x;
        double val22=-(1.0/5.0)*x*x*x*x*x+(1.0/4.0)*p*x*x*x*x+(1.0/4.0)*(p-h)*x*x*x*x-(1.0/3.0)*p*(p-h)*x*x*x;

        val2= val2 - val22;

        return val2-val1;
    }

    // integrate (x-(p-h))²
    double integral_CenterL(IndexDimension& cell, int d) const{
        double p = coord[d];
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p-h)*x*x*x*x+(1.0/3.0)*(p-h)*(p-h)*x*x*x;
        val1 = val1 - val12;


        x = x+h;
        double val2=(1.0/3.0)*x*x*x-(p-h)*x*x+(p-h)*(p-h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p-h)*x*x*x*x+(1.0/3.0)*(p-h)*(p-h)*x*x*x;
        val2= val2 - val22;

        return val2-val1;

    }

    // integrate (-x+(p+h))²*(1-x²)
    double integral_CenterR(IndexDimension& cell, int d) const{

        double p = coord[d];
        double h=meshwidth[d];
        double x = cell.coordinate(d);


        double val1= (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*(p+h)*(p+h)*x*x*x;
        val1 = val1 - val12;

        x = x+h;
        double val2=  (1.0/3.0)*x*x*x-(p+h)*x*x+(p+h)*(p+h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(2.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*(p+h)*(p+h)*x*x*x;
        val2= val2 - val22;
        return val2-val1;

    }


    // integrate (-x+(p+h))*(x-p)*(1-x²)
    double integral_right(IndexDimension& cell, int d) const{
        double p = coord[d];
        double h=meshwidth[d];
        double x = cell.coordinate(d);

        double val1=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
        double val12=(1.0/5.0)*x*x*x*x*x-(1.0/4.0)*p*x*x*x*x-(1.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*p*(p+h)*x*x*x;
        val1 = val1 + val12;

        x = x+h;
        double val2=-(1.0/3.0)*x*x*x+(1.0/2.0)*p*x*x+(1.0/2.0)*(p+h)*x*x-p*(p+h)*x;
        double val22=(1.0/5.0)*x*x*x*x*x-(1.0/4.0)*p*x*x*x*x-(1.0/4.0)*(p+h)*x*x*x*x+(1.0/3.0)*p*(p+h)*x*x*x;
        val2= val2 + val22;
        return val2-val1;

    }


private:
    double localValue(const IndexDimension &center ,IndexDimension& cell,const MultiDimCompass& mc)const{


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {
            if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d))
                value *= integral_CenterR(cell, d);
            if (mc.getRichtung(d) == Mitte && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_CenterL(cell, d);
            if (mc.getRichtung(d) == Links && !(cell.getIndex(d) == center.getIndex(d)))
                value *= integral_left(cell, d);
            if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d))
                value *= integral_right(cell, d);

            value = exponent[d]*exponent[d]*value;
        }




        return value;};











    Depth T;
    double meshwidth[DimensionSparseGrid];
    double exponent[DimensionSparseGrid];
    double coord[DimensionSparseGrid];

    PoissonStencil poisson;

};



class Poisson : public StencilTemplate{
public:
    Poisson(AdaptiveSparseGrid& grid_): StencilTemplate(grid_){};

    TypeMatrixVectorMultiplication getTypeMatrixVectorMultiplication(){
        return StencilOnTheFly;
    }


    inline double localValue(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc)const{


        double value =0.0;
        double ddx;
        for(int dx=0; dx < DimensionSparseGrid; dx++) {
            ddx=1.0;
            if (mc.getRichtung(dx) == Mitte && cell.getIndex(dx) == center.getIndex(dx)){

                ddx *= exponent[dx];
                ddx *= Integral_Mass(center,cell,mc,dx);

                value += ddx;

            }

            if (mc.getRichtung(dx) == Mitte && !(cell.getIndex(dx) == center.getIndex(dx))){

                ddx *= exponent[dx];
                ddx *= Integral_Mass(center,cell,mc,dx);

                value += ddx;

            }

            if (mc.getRichtung(dx) == Links && !(cell.getIndex(dx) == center.getIndex(dx))){

                ddx *= -1.0*exponent[dx];
                ddx *= Integral_Mass(center,cell,mc,dx);

                value += ddx;

            }

            if (mc.getRichtung(dx) == Rechts && cell.getIndex(dx) == center.getIndex(dx)){

                ddx *= -1.0*exponent[dx];
                ddx *= Integral_Mass(center,cell,mc,dx);

                value += ddx;

            }

        }


        return value;}

    inline double Integral_Mass(const IndexDimension &center ,const IndexDimension& cell, const MultiDimCompass &mc, int D)const{


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {

            if(D!=d) {
                if (mc.getRichtung(d) == Mitte && cell.getIndex(d) == center.getIndex(d)){


                    value *= integral_CenterR(center,cell, d);

                    // links links
                    value = exponent[d] * exponent[d] * value;

                }
                else if (mc.getRichtung(d) == Mitte && (cell.getIndex(d) != center.getIndex(d))){
                    value *= integral_CenterL(center,cell, d);

                    // rechts rechts
                    value = exponent[d] * exponent[d] * value;

                }

                else if (mc.getRichtung(d) == Links && (cell.getIndex(d) != center.getIndex(d))){
                    value *= integral_left(center,cell, d);

                    //
                    value = exponent[d] * exponent[d] * value;

                }

                else if (mc.getRichtung(d) == Rechts && cell.getIndex(d) == center.getIndex(d)){
                    value *= integral_right(center,cell, d);
                    value = exponent[d] * exponent[d] * value;

                }




            }
        }




        return value;
    }


    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){

        double h[DimensionSparseGrid];

        for(int i=0;i<DimensionSparseGrid;++i) h[i] =cell.getRightBoundary(i) - cell.getLeftBoundary(i);

/*        double sum=0.0;
        for(int i=0;i<DimensionSparseGrid;++i) {  // int du/dxi * dv/dxi  d(x1...xd)
            double prod=1.0;
            for(int j=0;j<DimensionSparseGrid;++j) { // integral factor in direction j
                if(i==j) {           // factor: int du/dxi * dv/dxi dxi
                    if(dirP.getDirection(j)==dirQ.getDirection(j)) prod =   prod / h[j];
                    else           prod = - prod / h[j];
                }
                else {               // factor: int du/dxi * dv/dxi dxj  == int u*v dxj
                    if(dirP.getDirection(j)==dirQ.getDirection(j)) prod =  prod * (1.0/3.0) * h[j];
                    else           prod =  prod * (1.0/6.0) * h[j];
                }
            }
            sum = sum + prod;
        }
        return sum;*/
        double value =0.0;
        double ddx;
        for(int dx=0; dx < DimensionSparseGrid; dx++) {
            ddx=1.0;
            if (dirP.getDirection(dx)==dirQ.getDirection(dx)){


                ddx *= exponent[dx];
                ddx *= Integral_Mass(cell,dirP,dirQ,dx);


                value += ddx;


            } else {


                ddx *= -1.0*exponent[dx];
                ddx *= Integral_Mass(cell,dirP,dirQ,dx);

                value += ddx;


            }

        }


        return value ;   // sum = sum_i int du/dxi * dv/dxi  d(x1...xd)
    }


    inline double Integral_Mass(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ, int D) {


        double value = 1.0;
        for(int d=0; d<DimensionSparseGrid; d++) {

            if(D!=d) {
                if (dirP.getDirection(d)==dirQ.getDirection(d)){
                    if(dirP.getDirection(d)==Right) value *= integral_RR(cell, d);
                    if(dirP.getDirection(d)==Left)  value *= integral_LL(cell,d);


                    value = exponent[d] * exponent[d] * value;

                }
                else {
                    value *= integral_LR(cell,d);
                    value = exponent[d] * exponent[d] * value;
                }
            }
        }
        return value;
    }



};




template<class StencilA,class StencilB>
class ADD:public StencilTemplate{
public:
     ADD(StencilA stencilA, StencilB stencilB,AdaptiveSparseGrid& grid_):StencilTemplate(grid_),stencilP(stencilA), stencilM(stencilB) {};
     virtual inline double returnValue(const IndexDimension &Index, const MultiDimCompass &mc){

        double val = 0.0;

        CellIterator cells(mc,stencilM.T,Index);


        for(cells.start(); cells.goon();++cells){
            IndexDimension cell = cells.getCell();
            if(stencilM.cellInGrid(cell)) {


                val += stencilM.localValue(Index, cell, mc);
                val += stencilP.localValue(Index, cell, mc);



            }

        }





        return val;



    };

    inline void initialize(Depth &T_) {

        stencilP.initialize(T_);
        stencilM.copyInitialization(stencilP);



    };

/*
   inline void applyStencilOnCell(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){
        stencilM.applyStencilOnCell(cell,input,output);
        stencilP.applyStencilOnCell(cell,input,output);


    }
*/
    inline double integration(CellDimension& cell, CellIndexDirection& dirP, CellIndexDirection& dirQ){

       return stencilP.integration(cell,dirP,dirQ)+stencilM.integration(cell,dirP,dirQ);
    }
/* inline void applyStencilOnCell(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){

        for(CellIndexIterator outerIter(&cell); outerIter.goon(); ++outerIter) {


            IndexDimension p = outerIter.getIndex();

            unsigned long kp;
            bool occP = output.getSparseGrid()->occupied(kp, p);
            CellIndexDirection dirP = outerIter.getCellIndexDirection();


            if (occP) {

                for (CellIndexIterator innerIter(outerIter); innerIter.goon(); ++innerIter) {


                    IndexDimension q = innerIter.getIndex();
                    if (q.isNotAtBoundary()) {

                        CellIndexDirection dirQ = innerIter.getCellIndexDirection();

                        unsigned long kq;

                        bool occQ = output.getSparseGrid()->occupied(kq, q);
                        if (occQ) {

                            double val = stencilM.integration(cell, dirP, dirQ)+stencilP.integration(cell, dirP, dirQ);
                            if (p == q) {
                                val = val * input.getValue(kp);

                                //#pragma omp critical
                                output.addToValue(kq, val);
                            } else {
                                double val_p = val * input.getValue(kp);
                                //#pragma omp critical
                                output.addToValue(kq, val_p);

                                double val_q = val * input.getValue(kq);
                                //#pragma omp critical
                                output.addToValue(kp, val_q);
                            }
                        }

                    }

                }

            }
        }









    }*/
/*    inline void applyStencilOnCell_BinaryIterator(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){
        stencilM.applyStencilOnCell_BinaryIterator(cell,input,output);
        stencilP.applyStencilOnCell_BinaryIterator(cell,input,output);
    }*/
/*    inline void applyStencilOnCell_MPI_OMP(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){
        stencilM.applyStencilOnCell_MPI_OMP(cell,input,output);
        stencilP.applyStencilOnCell_MPI_OMP(cell,input,output);


    }*/
/*    inline void applyStencilOnCell_MPI_OMP_nosymmetry(CellDimension& cell,VectorSparseG& input, VectorSparseG& output){
        stencilM.applyStencilOnCell_MPI_OMP_nosymmetry(cell,input,output);
        stencilP.applyStencilOnCell_MPI_OMP_nosymmetry(cell,input,output);


    }*/
private:

    StencilA stencilP;
    StencilB stencilM;

};



#endif //SGRUN_STENCIL_H
