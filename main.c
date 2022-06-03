static char help[] = "Calculate the smallest eigenvalue and corresponding eigenvector of the matrix.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            y, z, z_new;
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      y_norm,y_norm_new,error,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of vec y */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3],rstart,rend,nlocal,its;
  PetscScalar    zero = 0.0,one = 1.0,value[3],lambda;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Inverse power iteration for a matrix A to calculate the smallest
	 eigenvalue and the corresponding eigenvector of matrix A
	 A * y = z
	 z_new = y/y_norm
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&z);CHKERRQ(ierr);
  ierr = VecSetSizes(z,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(z);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&z_new);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&y);CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */
  ierr = VecGetOwnershipRange(z,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(z,&nlocal);CHKERRQ(ierr);

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
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Set z0
  */
  ierr = VecSet(z,zero);CHKERRQ(ierr);
  i = 0;
  ierr = VecSetValues(z,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(z);CHKERRQ(ierr);

  //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
                      Inverse power iteration
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  its=0;y_norm=0.0;
  do{
  its+=1;
  /*
     Solve linear system
   */
  ierr = KSPSolve(ksp,z,y);CHKERRQ(ierr);  

  ierr = VecNorm(y,NORM_2,&y_norm_new);CHKERRQ(ierr);

  ierr = VecAXPBY(z_new,one/y_norm_new,zero,y);CHKERRQ(ierr);

  error = PetscAbsReal(y_norm - y_norm_new);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error of y_norm %g, Iterations %D\n",(double)error,its);CHKERRQ(ierr);

  ierr = VecCopy(z_new,z);CHKERRQ(ierr);
  y_norm = y_norm_new;

  }while(error > tol && its < 10000);

  ierr = VecDot(y,z,&lambda);CHKERRQ(ierr);	//the inverse of the smallest eigenvalue of A
  lambda = one/lambda;				//the smallest eigenvalue of A
  ierr = PetscPrintf(PETSC_COMM_WORLD,"lambda = %g\n",(double)lambda);CHKERRQ(ierr);
  //ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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
  ierr = VecDestroy(&z);CHKERRQ(ierr); ierr = VecDestroy(&z_new);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
 // ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

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
