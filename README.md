# Solve the one-dimensional heat flux problem.

COMPILE:
  1. Open Makefile and modify the PETSC_DIR to the directory where you intall the PETSc.
  2. Enter "make solve.out" in the command line to generate the executable file.

OPTIONS:

  -tol		RealNumber

    Set the tolerence when computing the steady state solution.(1.e-7 by default)

  -c		RealNumber 

    Set the parameter c in the governing equation.(1.0 by default)

  -rho		RealNumber

    Set the parameter rho in the governing equation.(1.0 by default)

  -dx		RealNumber

    Set the spatial resolution dx.(0.01 by default)

  -dt		RealNumber

    Set the temporal resolution dt in seconds.(1.e-5 by default)

    No restriction when implicit Euler method is applied.

    dt should be smaller than (rho*c*dx^2/2) when explicit Euler method is applied.

  -t		RealNumber

    Set the time t in seconds to compute the transient solution at time t.(3600.0 by default)

    If the restart option is turned on, the computed transient solution is t seconds after the recorded solution in "result.h5"

  -l		RealNumber

    Set the parameter l in the governing equation.(1.0 by default)

  -Euler	IntNumber

    Set the method of Euler.(1 by default)

    0 : explicit Euler method

    1 : implicit Euler method

  -restart	IntNumber

    Set whether to restart from the exsiting file "result.h5".(0 by default)

    0 : no restart, compute from the time t = 0s

    1 : restart from the time t recorded in the file "result.h5"

  -state	IntNumber

    Set the state of the solution you want to compute.(1 by default)

    0 :	transient state solution

    1 : steady state solution
