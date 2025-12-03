


add_library(
        Applications




        cg_method.cpp cg_method.h

        power_method.h

        norms.cpp   norms.h

        condition.cpp condition.h

        PDE.cpp PDE.h

        H1_Norm.cpp
        H1_Norm.h
)

if (MPI_FOUND)
    target_link_libraries(Applications MPI::MPI_CXX )
else (MPI_FOUND)
    target_link_libraries(Applications)
endif (MPI_FOUND)

