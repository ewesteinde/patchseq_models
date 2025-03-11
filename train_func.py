import os
from IPython.display import clear_output

import matplotlib.pyplot as plt
import numpy as np
from jaxley.optimize.transforms import ParamTransform
import jax.numpy as jnp
import optax
import random
import time
from jax import jit, vmap, value_and_grad

from train_utils import Dataset, initialize_parameters, create_step_lr_scheduler
from loss_funcs import batched_sim_step, loss_fun_step, loss_fun_step_passive, simulate_step
from plot_utils import plot_parameter_trajectories


grad_fn_batched = jit(value_and_grad(loss_fun_step, argnums=1, has_aux = True), static_argnames=('cell', 'i_delay', 'i_dur', 'dt', 't_max', 'windows', 'bounds'))
grad_fn_batched_passive = jit(value_and_grad(loss_fun_step_passive, argnums=1, has_aux = True), static_argnames=('cell', 'i_delay', 'i_dur', 'dt', 't_max', 'windows', 'bounds'))
batched_sim_step = vmap(simulate_step, in_axes=(None, None, 0, None, None, None, None))
def make_hashable(bounds):
    return tuple(tuple(sorted(d.items())) for d in bounds)

def batched_stepCurrent_training(cell, bounds, data, windows, train_type = 'active', random_seeds = [1, 5, 10, 15], batches = 5, 
                                 batch_size = 5, steps = 50, learning_rate = 0.1, 
                                 beta = 0.8, required_loss = 5, patience_max = 50, 
                                 plot_num = 5, dataset_seed = 1, savedir = '/allen/programs/mindscope/workgroups/realistic-model/elena.westeinde/patchseq/patchseq_models/test_results'):

    transform = ParamTransform(bounds)            
    dataloader = Dataset(data)
    hashable_bounds = make_hashable(bounds)

    colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))
    
    total_sims = 0                       # Counter for total simulations run
    #beta = 0.8                     # Gradient normalization power factor
    plot_count = 0 

    lr_schedule = create_step_lr_scheduler(learning_rate, learning_rate/5, 1000)
  
    for random_seed in random_seeds:
        random.seed(random_seed)
        params = cell.get_parameters()
        params_init = initialize_parameters(bounds, random_seed=random_seed)
        print(params_init)
        opt_params =  transform.inverse(params_init)
        best_batch_params = opt_params
        optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm= 1),  # Add gradient clipping
        optax.adam(learning_rate=lr_schedule))
        opt_state = optimizer.init(opt_params)
    
        step_count = 0
        step_list = []
        train_losses = []
        train_params = {}
        all_losses = []
        grad_vals = []
        best_loss = 10000.0
    
        # make a empty metric dictionary with a key for each batch number
        batch_metrics = {batch: {'total_loss': [], 'losses': [], 'params': [], 'target': [], 'current': []} for batch in range(batches)}
        train_params[0] = transform.forward(opt_params)
        for batch in range(batches):
            # Stores loss trajectories for each batch
            
            batch_data, batch_current, batch_target, batch_amps = dataloader.get_batch(batch_size = batch_size, shuffle=True, seed=dataset_seed)
            
            batch_metrics[batch]['target'] = batch_target
            batch_metrics[batch]['current'] = batch_current

            batch_data[0]['input']['i_delay']
            i_delay = batch_data[0]['input']['i_delay']
            i_dur = batch_data[0]['input']['i_dur']
            dt = batch_data[0]['input']['dt']
            t_max = batch_data[0]['input']['t_max']
            
            cell.set('v', np.median(batch_target[:,0])) # set initial voltage to first time point of target voltage
            print('v', np.median(batch_target[:,0]))
            cell.set('Leak_eLeak', np.median(batch_target[:,0])) # set initial leak reversal potential to first time point of target voltage
            print('Leak_eLeak', np.median(batch_target[:,0]))
            cell.init_states()
            
            output_init = batched_sim_step(cell, params_init, batch_amps, i_delay, i_dur, dt, t_max)
            jax_v_init = np.array(output_init[:,0,:])

            patience = 0
            
            
            best_batch_loss = 10000.0
            for step in range(steps):
                step_t0 = time.time()
                # if set bounds on trainable params will need to transform in & out of constrained space
                params = transform.forward(opt_params)
                if train_type == 'active':
                    loss, grad_val = grad_fn_batched(cell, params, batch_target, batch_amps, i_delay, i_dur, dt, t_max, windows, hashable_bounds)
                else:
                    loss, grad_val = grad_fn_batched_passive(cell, params, batch_target, batch_amps, i_delay, i_dur, dt, t_max, windows, hashable_bounds)
                loss_val = loss[0]
                losses = loss[1]
                
                train_losses.append(loss_val)
                
                if jnp.isnan(loss_val): 
                    print("NAN loss")
                    break

                step_list.append(step_count)
                total_sims += 1
                
                    
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = opt_params

                if loss_val < best_batch_loss:
                    best_batch_loss = loss_val
                    best_batch_params = opt_params
                    patience = 0
                else:
                    patience += 1
                
                # Stop early if loss is below required threshold
                if loss_val < required_loss or patience > patience_max: #required_losses[iteration]:
                    opt_params = best_batch_params
                    break
                
                updates, opt_state = optimizer.update(grad_val, opt_state, opt_params)
                opt_params = optax.apply_updates(opt_params, updates)
                train_params[step + 1] = transform.forward(opt_params)
                grad_vals.append(grad_val)
                all_losses.append(losses)
    
                batch_metrics[batch]['total_loss'].append(loss_val)
                batch_metrics[batch]['losses'].append(losses)
                batch_metrics[batch]['params'].append(transform.forward(opt_params))
                
                print(f"Step time: {time.time() - step_t0:.4f}, Patience: {patience}")
                print(f"loss in epoch {step}: {loss_val:.4f}")
                print(f"All losses: {losses}")
                
                if step % plot_num == 0:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                    _ = ax.plot(train_losses, c="k")
                    plt.show()
                    
                    params = transform.forward(opt_params)
                    batched_output = batched_sim_step(cell, params, batch_amps, i_delay, i_dur, dt, t_max)
                    jax_v = batched_output[:,0,:]

                    
                    if batch_size <= 5:
                        fig, axs = plt.subplots(2, batch_size, figsize=(3*batch_size, 6))
                        for i in range(batch_size):
                            # Plot voltage for this batch (top row)
                            axs[0,i].plot(batch_target[i], color=colors[i], alpha=0.6, linewidth=2, label='BMTK')
                            darker_color = np.array(colors[i])
                            darker_color[:3] *= 0.7
                            axs[0,i].plot(jax_v[i, 2:], linestyle="--", color=darker_color, linewidth=0.5, label='JAX')
                            axs[0,i].set_title(f'Batch {i} - Voltage')
                            
                            
                            # Plot initial state for this batch (bottom row)
                            axs[1, i].plot(jax_v_init[i,:], color=colors[i], alpha=0.6, linewidth=2, label='Initial')
                            darker_color = np.array(colors[i])
                            darker_color[:3] *= 0.7
                            axs[1, i].plot(jax_v[i,:], linestyle="--", color=darker_color, linewidth=0.5, label='JAX')
                            axs[1, i].set_title(f'Batch {i} - Initial State')
                            
                            # # Add legend to the first column only to avoid cluttering
                            # if i == 0:
                            #     axs[0, i].legend()
                            #     axs[1, i].legend()

                        # Add y-labels to the leftmost column
                        axs[0, 0].set_ylabel('Voltage (mV)')
                        axs[1, 0].set_ylabel('Voltage (mV)')

                        # Add x-labels to the bottom row
                        for i in range(batch_size):
                            axs[1, i].set_xlabel('Time Steps')

                        plt.tight_layout()
                        plt.show()
                    else:
                        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
                        for i in range(batch_size):
                            _ = ax[0].plot(batch_target[i], color=colors[i], alpha=0.5, linewidth=2)
                            darker_color = np.array(colors[i])
                            darker_color[:3] *= 0.7
                            _ = ax[0].plot(jax_v[i, 2:], linestyle="--", color=darker_color, alpha=0.7, linewidth=1)

                            _ = ax[1].plot(jax_v_init[i], color=colors[i], alpha=0.6, linewidth=2)
                            darker_color = np.array(colors[i])
                            darker_color[:3] *= 0.7
                            _ = ax[1].plot(jax_v[i, 2:], linestyle="--", color=darker_color, alpha=0.7, linewidth=1)

                        plt.tight_layout()
                        plt.show()  
                    
                    plot_parameter_trajectories(train_params, reference_iter=0, log_scale_norm=True, figsize=(12, 4),
                                     title='Parameter Trajectories During Training') 
            
                    plt.show()
                    
                    plot_count += 1
                step_count += 1
            
            final_batch_params = transform.forward(best_batch_params)

            batched_output = batched_sim_step(cell, final_batch_params, batch_amps, i_delay, i_dur, dt, t_max)

            jax_v = batched_output[:,0,:]

            #savedir = '/allen/programs/mindscope/workgroups/realistic-model/elena.westeinde/patchseq/patchseq_models/test_results/060325_active/batch3_lr0.1_activeLoss'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            file_name = f"windowsStat_fit_allLoss_0.1_{random_seed}"
                        
            
            if batch_size <= 5:
                fig, axs = plt.subplots(2, batch_size, figsize=(3*batch_size, 4))
                for i in range(batch_size):
                    # Plot voltage for this batch (top row)
                    axs[0,i].plot(batch_target[i], color=colors[i], alpha=0.6, linewidth=2, label='BMTK')
                    darker_color = np.array(colors[i])
                    darker_color[:3] *= 0.7
                    axs[0,i].plot(jax_v[i, 2:], linestyle="--", color=darker_color, linewidth=0.5, label='JAX')
                    axs[0,i].set_title(f'Batch {i} - Voltage')
                    
                    
                    # Plot initial state for this batch (bottom row)
                    axs[1, i].plot(jax_v_init[i], color=colors[i], alpha=0.6, linewidth=2, label='Initial')
                    darker_color = np.array(colors[i])
                    darker_color[:3] *= 0.7
                    axs[1, i].plot(jax_v[i, 2:], linestyle="--", color=darker_color, linewidth=0.5, label='JAX')
                    axs[1, i].set_title(f'Batch {i} - Calcium')
                    
                    # # Add legend to the first column only to avoid cluttering
                    # if i == 0:
                    #     axs[0, i].legend()
                    #     axs[1, i].legend()

                # Add y-labels to the leftmost column
                axs[0, 0].set_ylabel('Voltage (mV)')
                axs[1, 0].set_ylabel('Voltage (mV)')

                # Add x-labels to the bottom row
                for i in range(batch_size):
                    axs[1, i].set_xlabel('Time Steps')

                plt.tight_layout()
            else:
                fig, ax = plt.subplots(1, 2, figsize=(6, 4))
                for i in range(batch_size):
                    _ = ax[0].plot(batch_target[i], color=colors[i], alpha=0.5, linewidth=2)
                    darker_color = np.array(colors[i])
                    darker_color[:3] *= 0.7
                    _ = ax[0].plot(jax_v[i, 2:], linestyle="--", color=darker_color, alpha=0.7, linewidth=1)

                    _ = ax[1].plot(jax_v_init[i], color=colors[i], alpha=0.6, linewidth=2)
                    darker_color = np.array(colors[i])
                    darker_color[:3] *= 0.7
                    _ = ax[1].plot(jax_v[i, 2:], linestyle="--", color=darker_color, alpha=0.7, linewidth=1)

                plt.tight_layout()
            
            plt.savefig(os.path.join(savedir, "best_params" + file_name + ".png"),
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.2)
            
            plt.show()
            
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            _ = ax.plot(train_losses, c="k")
            
            plt.savefig(os.path.join(savedir, "best_params" + file_name + "_loss.png"),
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.2)
            
            plt.show()
            
            plot_parameter_trajectories(train_params, reference_iter=0, log_scale_norm=True, figsize=(12, 4),
                                     title='Parameter Trajectories During Training')
            
            plt.savefig(os.path.join(savedir, "params_trajectory" + file_name + ".png"),
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.2)
            
            plt.show()
    
    final_params = transform.forward(best_params)
    
    return cell, final_params, best_loss, batch_metrics


