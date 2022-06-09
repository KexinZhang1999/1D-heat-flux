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

    When the restart option is turned on, make sure the dx input is the same value as the one recorded in the Restart file.

  -dt		RealNumber

    Set the temporal resolution dt in seconds.(1.e-5 by default)

    No restriction when implicit Euler method is applied.

    dt should be smaller than (rho*c*dx^2/2) when explicit Euler method is applied.

  -t		RealNumber

    Set the time t in seconds to compute the transient solution t seconds after the initial condition.(10.0 by default)

    If the restart option is turned on, the computed transient solution is t seconds after the recorded condition in "result.h5"

  -l		RealNumber

    Set the parameter l in the governing equation.(1.0 by default)

  -Euler	IntNumber

    Set the method of Euler.(1 by default)

    0 : explicit Euler method

    1 : implicit Euler method

  -restart	IntNumber

    Set whether to restart from the exsiting file "result.h5".(0 by default)

    0 : no restart, compute from the time t = 0s

    1 : restart from the solution recorded in the file "result.h5" and reset the initial conditon to be the one recorded in the restart file.

  -state	IntNumber

    Set the state of the solution you want to compute.(1 by default)

    0 :	transient state solution

    1 : steady state solution

Result:

  The computed solution will be stored in the file "result.h5"

  It contains a dataset "temperature" in the root group, and a dataset "grid: dx, dt, t" in the group /grid.

  "temperature" is the computed temperature on the x-domain of (0,1).

  The boundary temperature is not involved in the dataset.

  "grid: dx, dt, t" contains only three values which is dx, dt and t.

  dx represents the spatial resolution when running the code.

  dt represents the temporal resolution when running the code.

  t represents the solution is t seconds after the initial condition.
