import torch

class AdjointHeatStep(torch.autograd.Function):
    """
    Custom Autograd Function for a single Implicit Euler Time Step using the Adjoint Method.
    
    Forward: Solves A * T_next = T_prev
    Backward: Solves A^T * lambda = grad_output, then computes VJP for k.
    
    Only the solved state T_{n+1} (one temperature volume per time step) is kept for the
    backward pass, so memory is independent of the number of Jacobi iterations, unlike
    backpropagating through the unrolled solver.
    """
    
    @staticmethod
    def forward(ctx, T_prev, k, simulator):
        """
        Args:
            T_prev: Temperature at step n
            k: Diffusivity field
            simulator: Reference to DifferentiableHeatSimulator instance
        Returns:
            T_next: Temperature at step n+1
        """
        # 1. Solve Forward: (I - dt*L) * T_next = T_prev
        # Note: simulator._implicit_step expects the "RHS source" to be T_prev.
        # Inside _implicit_step, it computes b = T_prev / dt. 
        # The system solved is (1/dt * I - L) * T_next = 1/dt * T_prev.
        
        # We perform the solve without tracking gradients (handled manually in backward)
        with torch.no_grad():
            T_next = simulator._implicit_step(T_prev, k)
        
        # 2. Save tensors for backward
        # We save T_next and k. T_prev is not needed for the gradient w.r.t k or T_next, 
        # but we need to return gradient w.r.t T_prev which is just lambda.
        ctx.save_for_backward(T_next, k)
        ctx.simulator = simulator
        
        return T_next

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: dL/dT_next (The 'lambda' from the future step + direct loss grad)
        Returns:
            grad_T_prev: dL/dT_prev (The 'lambda' for this step)
            grad_k: dL/dk
            grad_simulator: None
        """
        T_next, k = ctx.saved_tensors
        simulator = ctx.simulator
        
        # --- Step 1: Solve Adjoint System ---
        # The forward linear system is A * T_next = b, where A = (1/dt I - L) and b = 1/dt * T_prev.
        # We need to find dL/db = lambda.
        # The relationship is: A^T * lambda = grad_output.
        #
        # Symmetry Argument:
        # The discrete Laplacian L with harmonic mean is symmetric.
        # Identity I is symmetric.
        # Therefore A = (1/dt I - L) is symmetric. A^T = A.
        #
        # We can reuse the EXACT SAME solver from the forward pass!
        # We just need to handle the scaling factor.
        # The forward solver solves: A * x = (1/dt) * input_val
        # We want to solve: A * lambda = grad_output
        #
        # If we pass (grad_output * dt) into the solver:
        # A * lambda = (1/dt) * (grad_output * dt) = grad_output.
        # Correct!
        
        with torch.no_grad():
            # Solve for lambda (adjoint variable)
            # Input to _implicit_step is treated as the "Temperature state" from previous step
            adj_lambda = simulator._implicit_step(grad_output * simulator.dt, k)
            
        # grad_T_prev is dL/dT_prev.
        # Since T_next = A^{-1} * (1/dt * T_prev)
        # dL/dT_prev = dL/dT_next * dT_next/dT_prev
        #            = grad_output * A^{-1} * (1/dt)
        #            = lambda * (1/dt)
        grad_T_prev = adj_lambda * simulator.inv_dt
        
        # --- Step 2: Compute Gradient w.r.t k (Diffusivity) ---
        # We need dL/dk = dL/dT_next * dT_next/dk
        # Implicit function theorem / Adjoint method:
        # dL/dk = - lambda^T * (dA/dk * T_next - db/dk)
        # Here b = T_prev/dt (independent of k), so db/dk = 0.
        # dL/dk = - lambda^T * (dA/dk * T_next)
        #
        # A = (1/dt * I - L(k))
        # dA/dk = - d(L(k))/dk
        #
        # So: dL/dk = lambda^T * (d(L(k))/dk * T_next)
        #           = lambda^T * d(L(k) * T_next) / dk
        #
        # We can compute (L(k) * T_next) using a local forward pass (apply_operator)
        # and then use autograd.grad to get the VJP.
        
        with torch.enable_grad():
            k_detached = k.detach().requires_grad_(True)
            T_next_detached = T_next.detach()
            
            # Compute only the Flux/Laplacian term: L(k) * T
            # We need a function that applies L without the 1/dt term.
            # Alternatively, we can use apply_system_matrix which computes (1/dt*I - L)*T
            # Let resid = A(k) * T_next
            # d(resid)/dk = - d(L)/dk * T_next
            # We want: lambda^T * (d(L)/dk * T_next)
            #
            # If we compute vjp of (A(k) * T_next) against lambda:
            # vjp = lambda^T * dA/dk = lambda^T * (-dL/dk) = - grad_we_want
            # So grad_k = - vjp
            
            # Compute A * T_next
            system_output = simulator.apply_system_matrix(T_next_detached, k_detached)
            
            # Compute VJP
            # grad_k = - d(system_output)/dk * lambda
            grad_k = torch.autograd.grad(
                outputs=system_output,
                inputs=k_detached,
                grad_outputs=adj_lambda, # This is lambda
                retain_graph=False
            )[0]
            
            # The negative sign comes from the implicit function theorem derivation:
            # F(T, k) = A(k)T - b = 0
            # dT/dk = - A^{-1} * dA/dk * T
            # dL/dk = grad_output * dT/dk = - (grad_output A^{-1}) * dA/dk * T
            #       = - lambda * dA/dk * T
            #
            # The autograd.grad computes (lambda * dA/dk * T).
            # So we need to negate it.
            grad_k = -grad_k
            
        return grad_T_prev, grad_k, None
