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

int main(int argc,char **args)
{
  Vec            u,u_next,f;       /* vector u for temperature, vector f for the force (sine function) */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of vector u */
  PetscErrorCode ierr;
  PetscInt       i,n = 99,col[3],rstart,rend,nlocal,its;	/* n is the length of vector u, 99 in this case*/
  PetscScalar    zero = 0.0,one = 1.0,value[3],c=1.0,rho=1.0,dt=0.00001,l,lambda,diag;
		/* c and rho are the parameters in the heat equation, l is the parameter in the sine function */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Get the value of c, rho, dt, l for particular options */
  ierr = PetscOptionsGetReal(NULL,NULL,"-c",&c,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-rho",&rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);

  lambda = rho * c / dt;
//if explicit
  diag = 20000. - lambda;
//if implicit
  diag = 20000. + lambda;	/* diag is the diagonal element of matrix A */
//if explicit, check lambda,assertion
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
    i      = 0; col[0] = 0; col[1] = 1; value[0] = diag; value[1] = -10000.0;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -10000.0; value[1] = diag;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -10000.0; value[1] = diag; value[2] = -10000.0;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Initialize u to be zero vector */
  ierr = VecSet(u,zero);CHKERRQ(ierr);
  
  /* Assemble the vector f */
  for (i=rstart; i<rend;i++)
  {
    value[0] = sin(0.01*l*3.1415926*(i+1));
    ierr = VecSetValues(f,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

  //ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	If implicit Euler method is used       
	Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Iteration on vector u
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
//if implicit
  its=0;
  do{
  its+=1;
  /* Solve linear system A * u_next = lambda * u + f */
  ierr = VecAYPX(u,lambda,f);CHKERRQ(ierr);		// u = f + lambda * u
  ierr = KSPSolve(ksp,u,u_next);CHKERRQ(ierr);		// A * u_next = u  

  ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);		// u = u - u_next
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);		// compute the norm of (u - u_next)

  //write to HDF5 every 10 iterations

  ierr = VecCopy(u_next,u);CHKERRQ(ierr);		// copy u_next to u, go to next time step

  }while(norm > tol && its < 100000000);

// if explicit
  its=0;
  /* Compute u_next = -1/lambda * A * u + 1/lambda * f */
  ierr = MatScale(A,-1.0/lambda);CHKERRQ(ierr);		// A = -1.0/lambda * A
  ierr = VecScale(f,1.0/lambda);CHKERRQ(ierr);		// f = 1/lambda * f
  /* The algorithm becomes u_next = A * u + f */
  do{
  its+=1;
  ierr = MatMultAdd(A,u,f,u_next);CHKERRQ(ierr);	// u_next = A * u + f

  ierr = VecAXPY(u,-1.0,u_next);CHKERRQ(ierr);          // u = u - u_next
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);         // compute the norm of (u - u_next)

  //write to HDF5 every 10 iterations
 
  ierr = VecCopy(u_next,u);CHKERRQ(ierr);               // copy u_next to u, go to next time step
 
  }while(norm > tol && its < 100000000);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  *//*
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }*/

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&u_next);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

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
