static char help[] = "Solve the one-dimensional heat flux problem.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>
#include <petscviewerhdf5.h>
#include <assert.h>
#define pi acos(-1.0)
#define FILE "result.h5"

int main(int argc,char **args)
{
  Vec            u,u_next,u_exact,f,grid;  /* vector u for temperature, vector f for the force (sine function) */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscViewer	 viewer;	   /* for HDF5 input/output */
  PetscErrorCode ierr;
  PetscInt       i,n,col[3],rstart,rend,nlocal,its=0,its_max,restart=0,Euler=1,state=1; 
		 /* n is the length of vector u, 1/dx-1 in this case
 		    Euler: 1 for implicit (default) and 0 for explicit
		    restart: 0 for no restart (default) and 1 for restart
		    state: 0 for transient state solution and 1 for steady state solution (default)
		  */
  PetscReal      norm,tol=1.e-7,error;
		/* norm is the NORM_2 of (u-u_next), tol defines the tolerence of steady state solution,
 		   error is NORM_INFINITY of (u-u_exact)*/
  PetscScalar    value[3],c=1.0,rho=1.0,dx=0.01,dt=0.00001,l=1.0,lambda,diag,t=3600.;
		/* c and rho are the parameters in the heat equation, l is the parameter in the sine function */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Get the value of c, rho, dt, l and the method for Euler for particular options */
  ierr = PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-c",&c,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-rho",&rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dx",&dx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-t",&t,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Euler",&Euler,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-state",&state,NULL);CHKERRQ(ierr);

  /* Assert that c, rho, dt, l are positive, Euler method is explicit or implicit */
  assert(tol>0.0);
  assert(c>0.0);
  assert(rho>0.0);
  assert(dx>0.0);
  assert(dt>0.0);
  assert(t>0.0);
  assert(l>0.0);
  assert(Euler==0||Euler==1);
  assert(restart==0||restart==1);
  assert(state==0||state==1);

  its_max = t / dt;
  n = 1.0/dx - 1;
  lambda = rho * c / dt;
  if(Euler==0)
  { //if explicit
    //assert(lambda>(2.0/dx/dx));	/* assert that lambda>2/dx/dx when explicit Euler method is chosen */
    diag = 2./dx/dx - lambda;	/* diag is the diagonal element of matrix A */
  }else if(Euler==1)
  { //if implicit
    diag = 2./dx/dx + lambda;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         To get the temperature vector at time t, there are two methods.

	 One is explicit Euler method, which is conditionally stable.
	 The selection of dt should be checked.
	 u_next = -1 / lambda * A * u + 1 / lambda * f

	 The other method is implicit Euler method, which is
	 unconditionally stable. dt can be unchecked.
	 A * u_next = lambda * u + f
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&f);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&u_next);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&u_exact);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "temperature");CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&grid);CHKERRQ(ierr);
  ierr = VecSetSizes(grid,PETSC_DECIDE,3);CHKERRQ(ierr);	// vec grid contains 3 values:dx,dt,t
  ierr = VecSetFromOptions(grid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)grid, "grid:dx,dt,t");CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */
  ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.

     We pass in nlocal as the "local" size of the matrix to force it
     to have the same parallel layout as the vector created above.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */
  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = diag; value[1] = -1.0/dx/dx;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0/dx/dx; value[1] = diag;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0/dx/dx; value[1] = diag; value[2] = -1.0/dx/dx;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Initialize u to be exp(x) */
  for (i=rstart; i<rend;i++)
  {
    value[0] = exp(dx*(i+1));
    ierr = VecSetValues(u,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  
  /* Assemble the exact solution when t tends to infinity */
  /* u_exact = sin(l*pi*x)/l/l/pi/pi - x * sin(l*pi)/l/l/pi/pi where x = (i+1)*dx */
  for (i=rstart; i<rend;i++)
  {
    value[0] = sin(l*pi*dx*(i+1))/l/l/pi/pi - (i+1) * dx * sin(l*pi)/l/l/pi/pi;
    ierr = VecSetValues(u_exact,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u_exact);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u_exact);CHKERRQ(ierr);

  /* Assemble the vector f */
  for (i=rstart; i<rend;i++)
  {
    value[0] = sin(dx*l*pi*(i+1));
    ierr = VecSetValues(f,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

  /* Assemble the vector grid which contains 3 values, which are dx, dt and t */
  col[0] = 0; col[1] = 1; col[2] = 2;
  value[0] = dx; value[1] = dt; value[2] = 0.0;	//dx,dt,t=0
  ierr = VecSetValues(grid,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);

  if(!restart)
  {/* Create the h5 file if compute from the beginning*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,FILE,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  }else if(restart)
  {/* Open the h5 file if restart*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,FILE,FILE_MODE_UPDATE,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(u,viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);
    ierr = VecLoad(grid,viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);    
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	If implicit Euler method is used       
	Create the linear solver and set various options
	Do iteration on vector u
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if(Euler==1)
  {
  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Iteration on vector u */
  if(state==0){
  do{
  its+=1;
  /* Solve linear system A * u_next = lambda * u + f */
  ierr = VecAYPX(u,lambda,f);CHKERRQ(ierr);		// u = f + lambda * u
  ierr = KSPSolve(ksp,u,u_next);CHKERRQ(ierr);		// A * u_next = u  

  ierr = VecAXPBY(u,-1.0/lambda,1.0/lambda,f);CHKERRQ(ierr);    // return to the original u = 1/lambda * u - 1/lambda * f
  ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);          // u = u - u_next
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);         // compute the NORM_2 of u
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm   %g      iterations      %D\n",norm,its);CHKERRQ(ierr);

  ierr = VecCopy(u_next,u);CHKERRQ(ierr);               // copy u_next to u, go to next time step

  /* write to HDF5 every 10 iterations */
  i = 2; value[0] = 10.0*dt;
  if(its%10==0){
    ierr = VecView(u,viewer);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);

    ierr = VecSetValues(grid,1,&i,value,ADD_VALUES);CHKERRQ(ierr);     //Set the time grid[2]=grid[2]+10*dt
    ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);
    ierr = VecView(grid,viewer);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  }while(its < its_max);	// the iteration stops until its > t / dt when computing transient state solution
  }else if(state==1){
  do{
  its+=1;
  /* Solve linear system A * u_next = lambda * u + f */
  ierr = VecAYPX(u,lambda,f);CHKERRQ(ierr);             // u = f + lambda * u
  ierr = KSPSolve(ksp,u,u_next);CHKERRQ(ierr);          // A * u_next = u

  ierr = VecAXPBY(u,-1.0/lambda,1.0/lambda,f);CHKERRQ(ierr);	// return to the original u = 1/lambda * u - 1/lambda * f
  ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);		// u = u - u_next
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);		// compute the NORM_2 of u
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm   %g      iterations      %D\n",norm,its);CHKERRQ(ierr);

  ierr = VecCopy(u_next,u);CHKERRQ(ierr);               // copy u_next to u, go to next time step

  /* write to HDF5 every 10 iterations */
  i = 2; value[0] = 10.0*dt;
  if(its%10==0){
    ierr = VecView(u,viewer);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);

    ierr = VecSetValues(grid,1,&i,value,ADD_VALUES);CHKERRQ(ierr);     //Set the time grid[2]=grid[2]+10*dt
    ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);
    ierr = VecView(grid,viewer);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  }while(norm > tol);	// the iteration stops until the value of u get stable when computing steady state solution
  }
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Iteration on vector u when explicit Euler method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if(Euler==0)
  {
    if(state==0){
    /* Compute u_next = -1/lambda * A * u + 1/lambda * f */
    ierr = MatScale(A,-1.0/lambda);CHKERRQ(ierr);		// A = -1.0/lambda * A
    ierr = VecScale(f,1.0/lambda);CHKERRQ(ierr);		// f = 1/lambda * f
    /* The algorithm becomes u_next = A * u + f */
    do{
    its+=1;
    ierr = MatMultAdd(A,u,f,u_next);CHKERRQ(ierr);	// u_next = A * u + f

    ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);          // u = u - u_next
    ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);         // compute the NORM_2 of u
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm   %g      iterations      %D\n",norm,its);CHKERRQ(ierr);

    ierr = VecCopy(u_next,u);CHKERRQ(ierr);               // copy u_next to u, go to next time step

    /* write to HDF5 every 10 iterations */
    i = 2; value[0] = 10.0*dt;
    if(its%10==0){
      ierr = VecView(u,viewer);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);
      
      ierr = VecSetValues(grid,1,&i,value,ADD_VALUES);CHKERRQ(ierr);	//Set the time grid[2]=grid[2]+10*dt
      ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);
      ierr = VecView(grid,viewer);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } 
    }while(its < its_max);	// the iteration stops until its > t / dt when computing transient state solution
    }else if(state==1){
    /* Compute u_next = -1/lambda * A * u + 1/lambda * f */
    ierr = MatScale(A,-1.0/lambda);CHKERRQ(ierr);               // A = -1.0/lambda * A
    ierr = VecScale(f,1.0/lambda);CHKERRQ(ierr);                // f = 1/lambda * f
    /* The algorithm becomes u_next = A * u + f */
    do{
    its+=1;
    ierr = MatMultAdd(A,u,f,u_next);CHKERRQ(ierr);      // u_next = A * u + f

    ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);          // u = u - u_next
    ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);         // compute the NORM_2 of u
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm	%g	iterations	%D\n",norm,its);CHKERRQ(ierr);

    ierr = VecCopy(u_next,u);CHKERRQ(ierr);               // copy u_next to u, go to next time step

    /* write to HDF5 every 10 iterations */
    i = 2; value[0] = 10.0*dt;
    if(its%10==0){
      ierr = VecView(u,viewer);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);

      ierr = VecSetValues(grid,1,&i,value,ADD_VALUES);CHKERRQ(ierr);    //Set the time grid[2]=grid[2]+10*dt
      ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);
      ierr = VecView(grid,viewer);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
    }while(norm > tol);	// the iteration stops until the value of u get stable when computing steady state solution
    }
  }

  /* - - - - - - Write the final solution to HDF5 file- - - - - - - - - - */

  ierr = VecView(u,viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/grid");CHKERRQ(ierr);

  ierr = VecSetValues(grid,1,&i,value,ADD_VALUES);CHKERRQ(ierr);    //Set the time grid[2]=grid[2]+10*dt
  ierr = VecAssemblyBegin(grid);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(grid);CHKERRQ(ierr);
  ierr = VecView(grid,viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error between the computed numerical solution
		     and the analytical solution when t tends to inifity
  */
  ierr = VecAXPY(u_exact,-1.0,u);CHKERRQ(ierr);	// u_exact = u_exact - u
  ierr = VecNorm(u_exact,NORM_INFINITY,&error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"error: %g, dx: %g, dt: %g,iterations: %D\n",error,dx,dt,its);CHKERRQ(ierr);


  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&u_next);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&grid);CHKERRQ(ierr);
  ierr = VecDestroy(&u_exact);CHKERRQ(ierr);
  if(Euler==1){
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

// EOF
