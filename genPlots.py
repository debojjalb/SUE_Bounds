import matplotlib.pyplot as plt
import os
import pandas as pd
import random
from numpy.linalg import norm
from tqdm import tqdm
from helper import dict_difference
import numpy as np
import shutil

random.seed(42)

def PlotLinkFlowBound(stochasticUE, optimalLinkFlows, folder, neworkname):
    """
    Plots the bounds on link flows against the actual distance from the optimal link flows over iterations.
    """
    x_values = []

    bound_y1 = []
    bound_y2 = []

    bound_y1_rel = []
    bound_y2_rel = []

    optimal_link_flows = {link: flow for link, flow in optimalLinkFlows.items()}

    for iters in range(len(stochasticUE.linkFlowsOverIts)):
        link_flows = {link: flow  for link, flow in stochasticUE.linkFlowsOverIts[iters].items()} 

        link_flow_norm = norm(np.array(list(dict_difference(link_flows, optimal_link_flows).values())))
        predicted_bound = (stochasticUE.linkBoundOverIts[iters] ) 
        
        bound_y1.append(link_flow_norm)
        bound_y2.append(predicted_bound)
        x_values.append(iters)

        link_flow_norm = norm(np.array(list(dict_difference(link_flows, optimal_link_flows).values()))) / norm(np.array(list(optimal_link_flows.values())))
        predicted_bound = (stochasticUE.linkBoundOverIts[iters] ) / norm(np.array(list(optimal_link_flows.values())))

        bound_y1_rel.append(link_flow_norm)
        bound_y2_rel.append(predicted_bound)
        

    plt.plot(x_values, bound_y1_rel, label='Actual', linestyle='-', linewidth = 3)
    plt.plot(x_values, bound_y2_rel, label='Bound', linestyle='-', linewidth = 3)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Rel. distance', fontsize = 34)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/link flow bound.pdf')
    plt.close()

    # Plot with last 20 iterations
    plt.plot(x_values[-20:], bound_y1_rel[-20:], label='Actual', linestyle='-', linewidth = 3)
    plt.plot(x_values[-20:], bound_y2_rel[-20:], label='Bound', linestyle='-', linewidth = 3)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Rel. distance', fontsize=34)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xticks(np.arange(min(x_values[-20:]), max(x_values[-20:]) + 1, 10))
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/link flow bound last 20.pdf')
    plt.close() 

    # Now a log plot
    plt.plot(x_values, np.log(bound_y1), linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values, np.log(bound_y2), linestyle='-', label='Bound', linewidth = 3)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Log norm', fontsize=34)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/log link flow bound.pdf')
    plt.close()

    shutil.copy(f'{folder}/link flow bound.pdf', f'./Plots/paperPlots/{neworkname}_link flow bound.pdf')
    shutil.copy(f'{folder}/link flow bound last 20.pdf', f'./Plots/paperPlots/{neworkname}_link flow bound last 20.pdf')
    shutil.copy(f'{folder}/log link flow bound.pdf', f'./Plots/paperPlots/{neworkname}_log link flow bound.pdf')
    
def PlotTTBound(stochasticUE, ooptimalLinkCost, folder, neworkname):
    """
    Plots the bounds on travel time against the actual distance from the optimal travel time over iterations.
    """
    x_values = []
    bound_y1 = []
    bound_y2 = []
    
    bound_y1_rel = []
    bound_y2_rel = []


    optimal_link_costs = ooptimalLinkCost

    for iters in range(len(stochasticUE.linkCostsOverIts)):
        link_cost = stochasticUE.linkCostsOverIts[iters]
        link_cost_norm = norm(np.array(list(dict_difference(link_cost, optimal_link_costs).values())))
        predicted_bound = stochasticUE.ttboundsOverIts[iters]
       
        bound_y1.append(link_cost_norm)
        bound_y2.append(predicted_bound)
        x_values.append(iters)

        link_cost_norm = norm(np.array(list(dict_difference(link_cost, optimal_link_costs).values()))) / norm(np.array(list(optimal_link_costs.values())))
        predicted_bound = stochasticUE.ttboundsOverIts[iters] / norm(np.array(list(optimal_link_costs.values())))

        bound_y1_rel.append(link_cost_norm)
        bound_y2_rel.append(predicted_bound)

    plt.plot(x_values, bound_y1_rel, linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values, bound_y2_rel, linestyle='-', label='Bound', linewidth = 3)
    # plt.grid(True)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Rel. distance', fontsize=34)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/travel time bound.pdf')
    plt.close()

    # Plot last 20 iterations
    plt.plot(x_values[-20:], bound_y1_rel[-20:], linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values[-20:], bound_y2_rel[-20:], linestyle='-', label='Bound', linewidth = 3 )
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Rel. distance', fontsize=34)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xticks(np.arange(min(x_values[-20:]), max(x_values[-20:]) + 1, 10))
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/travel time bound last 20.pdf')
    plt.close()

    # Now a log plot
    plt.plot(x_values, np.log(bound_y1), linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values, np.log(bound_y2), linestyle='-', label='Bound', linewidth = 3)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Log norm', fontsize=34)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/log travel time bound.pdf')
    plt.close()

    shutil.copy(f'{folder}/travel time bound.pdf', f'./Plots/paperPlots/{neworkname}_travel time bound.pdf')
    shutil.copy(f'{folder}/travel time bound last 20.pdf', f'./Plots/paperPlots/{neworkname}_travel time bound last 20.pdf')
    shutil.copy(f'{folder}/log travel time bound.pdf', f'./Plots/paperPlots/{neworkname}_log travel time bound.pdf')

def PlotTSTTBound(stochasticUE, optimalTSTT, folder, neworkname):
    """
    Plots the bounds on TSTT against the actual distance from the optimal TSTT over iterations.
    """
    x_values = []
    actual_diff = []
    bound_diff = []
    actual_diff_rel = []
    bound_diff_rel = []

    optimal_TSTT = optimalTSTT
    for iters in range(2, len(stochasticUE.TSTTOverIts)):

        actual_difference = abs(stochasticUE.TSTTOverIts[iters] - optimal_TSTT) 
        predicted_bound = stochasticUE.tsttBoundOverIts[iters]
       
        # print("Actual difference", actual_difference, "Predicted bound", predicted_bound)
        actual_diff.append(actual_difference)
        bound_diff.append(predicted_bound)
        x_values.append(iters)

        actual_difference = abs(stochasticUE.TSTTOverIts[iters] - optimal_TSTT) / optimal_TSTT
        predicted_bound = stochasticUE.tsttBoundOverIts[iters] / optimal_TSTT
        actual_diff_rel.append(actual_difference)
        bound_diff_rel.append(predicted_bound)
    
    # print("TSTT Bound", list(bound_diff)[-10:])
    # print("TSTT Actual", list(actual_diff)[-10:])

    plt.plot(x_values, actual_diff_rel, linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values, bound_diff_rel, linestyle='-', label='Bound', linewidth = 3)
    plt.xlabel('Iterations', fontsize=34)
    plt.ylabel('Rel. distance', fontsize=34)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/TSTT_Bound.pdf')
    plt.close()

    # Plot last 20 iterations
    plt.plot(x_values[-20:], actual_diff_rel[-20:], linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values[-20:], bound_diff_rel[-20:], linestyle='-', label='Bound', linewidth =3)
    plt.xlabel('Iterations', fontsize=36)
    plt.ylabel('Rel. distance', fontsize=36)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xticks(np.arange(min(x_values[-20:]), max(x_values[-20:]) + 1, 10))
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/TSTT_Bound_last_20.pdf')
    plt.close()

    # Now a log plot
    plt.plot(x_values, np.log(actual_diff), linestyle='-', label='Actual', linewidth = 3)
    plt.plot(x_values, np.log(bound_diff), linestyle='-', label='Bound', linewidth = 3)
    plt.xlabel('Iterations', fontsize=36)
    plt.ylabel('Log difference', fontsize=36)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().yaxis.get_offset_text().set_fontsize(26)
    plt.tight_layout()
    plt.savefig(f'{folder}/log TSTT_Bound.pdf')
    plt.close()

    shutil.copy(f'{folder}/TSTT_Bound.pdf', f'./Plots/paperPlots/{neworkname}_TSTT_Bound.pdf')
    shutil.copy(f'{folder}/TSTT_Bound_last_20.pdf', f'./Plots/paperPlots/{neworkname}_TSTT_Bound_last_20.pdf')
    shutil.copy(f'{folder}/log TSTT_Bound.pdf', f'./Plots/paperPlots/{neworkname}_log TSTT_Bound.pdf')

def PlotBound(stochasticUE, optimalSol, folder, r, neworkname):
    """
    Plots the bounds on path flows against the actual distance from the optimal path flows over iterations.
    """
    for OD in tqdm(stochasticUE.currentPathFlowsOverIts):

        if OD in stochasticUE.od_with_no_assignments:
            continue

        curr_path_flow = stochasticUE.currentPathFlowsOverIts[OD]
        target_path_flow = stochasticUE.targetFlowOverIts[OD]
        lambdas = stochasticUE.lambdaMaxOverIts[OD]
        optimalPathFlow = optimalSol[OD]
        
        x_values = []
        bound = []
        actual = []
        h_star = optimalPathFlow

        for it in range(len(curr_path_flow) - 1):
            
            h_i = curr_path_flow[it]
            h_next = target_path_flow[it]
            x_values.append(it)

            bound.append(abs(lambdas[it]) * norm(np.array(list(dict_difference(h_next, h_i).values()))))
            actual.append(norm(np.array(list(dict_difference(h_star, h_i).values()))))
        
        # Plot the bound and actual distance from eqm.
        plt.plot(x_values, actual, label='Actual distance from eqm.', linewidth = 3)
        plt.plot(x_values, bound, label='Bound on distance from eqm.', linewidth=3)
        plt.xlabel('Iterations', fontsize=18)
        plt.ylabel('Norm', fontsize=18)
        plt.title('OD: {}'.format(OD), fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(f'{folder}/Bound/r{r}_bound matrix h bound for OD pair {OD}.pdf')
        plt.tight_layout()
        plt.close()

        # Log plot of the bound and actual distance from eqm.
        plt.plot(x_values[1:], np.log(actual)[1:], label='Actual distance from eqm.', linewidth = 3)
        plt.plot(x_values[1:], np.log(bound)[1:], label='Bound on distance from eqm.', linewidth=3)
        plt.xlabel('Iterations', fontsize=18)
        plt.ylabel('Log norm', fontsize=18) 
        plt.title('OD: {}'.format(OD), fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{folder}/LogBound/r{r}_log bound matrix h bound for OD pair {OD}.pdf')
        plt.close()
    
    all_OD = list(optimalSol.keys())
    all_OD = [OD for OD in all_OD if OD not in stochasticUE.od_with_no_assignments]

    try: 
        OD_sample = random.sample(all_OD, 16)

        # Plot 5x5 graphs of random OD pairs
        for _i in range(5):
            
            fig, axs = plt.subplots(4, 4, figsize=(15, 15))

            for i, OD in enumerate(OD_sample):

                row, col = divmod(i, 4)

                x_values = np.arange(1, len(actual) + 1)
                actual = np.array(actual)
                bound = np.array(bound)

                axs[row, col].plot(x_values, actual, linestyle='-', label='Actual', linewidth=3)
                axs[row, col].plot(x_values, bound, linestyle='-',label='Bound', linewidth=3)
                axs[row, col].set_title(f"OD: {OD}", fontsize=22)

                axs[row, col].set_xlabel('Iterations', fontsize=20)
                axs[row, col].set_ylabel('Norm', fontsize=20)
                axs[row, col].tick_params(axis='x', labelsize=20)
                axs[row, col].tick_params(axis='y', labelsize=20)

                if row == 0 and col == 3:
                    axs[row, col].legend(fontsize=15)
                # axs[row, col].grid(True)

            plt.tight_layout()
            plt.savefig(f'{folder}/Multiplots/Bound_{_i}.pdf')
            plt.close()

        shutil.copy(f'{folder}/Multiplots/Bound_{_i}.pdf', f'./Plots/paperPlots/{neworkname}_Bound_{_i}.pdf')

        # Plot 5x5 graphs of random OD pairs

        for _i in range(5):

            fig, axs = plt.subplots(4, 4, figsize=(15, 15))

            for i, OD in enumerate(OD_sample):

                row, col = divmod(i, 4)

                x_values = np.arange(1, len(actual) + 1)
                actual = np.array(actual)
                bound = np.array(bound)

                axs[row, col].plot(x_values, np.log(actual), linestyle='-', label='Actual', linewidth=3)
                axs[row, col].plot(x_values, np.log(bound), linestyle='-', label='Bound', linewidth=3)
                axs[row, col].set_xlabel('Iterations', fontsize=20)
                axs[row, col].set_ylabel('Log Norm', fontsize=20)
                axs[row, col].tick_params(axis='x', labelsize=20)
                axs[row, col].tick_params(axis='y', labelsize=20)
                # axs[row, col].grid(True
                
                axs[row, col].set_title(f"OD: {OD}", fontsize=22)
                if row == 0 and col == 3:
                    axs[row, col].legend(fontsize=15)

            plt.tight_layout()
            plt.savefig(f'{folder}/Multiplots/LogBound_{_i}.pdf')
            plt.close()
        
        shutil.copy(f'{folder}/Multiplots/LogBound_{_i}.pdf', f'./Plots/paperPlots/{neworkname}_LogBound_{_i}.pdf')
    except ValueError:
        OD_sample = []
        pass
    return OD_sample

def plotConvergence(stochasticUE, optimalSol, folder, r, networkname, OD_sample):
    """
    Plots the convergence of the bounds on path flows against the actual distance from the optimal path flows over iterations.
    """

    error_od = {}

    for OD in stochasticUE.currentPathFlowsOverIts:

        if OD in stochasticUE.od_with_no_assignments:
            continue

        curr_path_flow = stochasticUE.currentPathFlowsOverIts[OD]
        target_path_flow = stochasticUE.targetFlowOverIts[OD]
        lambdas = stochasticUE.lambdaMaxOverIts[OD]
        optimalPathFlow = optimalSol[OD]
        
        x_values = []
        bound = []
        actual = []
        convergence = []
        h_star = optimalPathFlow

        for it in range(len(curr_path_flow) - 1):
            
            h_i = curr_path_flow[it]
            h_next = target_path_flow[it]
            x_values.append(it)

            bound.append(abs(lambdas[it]) * norm(np.array(list(dict_difference(h_next, h_i).values()))))
            actual.append(norm(np.array(list(dict_difference(h_star, h_i).values()))))
            convergence.append((bound[-1] - actual[-1]))
        
        error_od[OD] = convergence

        # Plot the convergence
        plt.plot(x_values, convergence, label='Bound gap', linewidth = 3, linestyle='-')
        plt.xlabel('Iterations', fontsize=18)
        plt.ylabel('Norm', fontsize=18)
        plt.title('OD: {}'.format(OD), fontsize=18)
        # plt.grid(True)
        # plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'{folder}/Convergence/r{r}_convergence matrix h bound for OD pair {OD}.pdf')
        plt.close()
    
    try:
    # plot 5, 5x5 graphs of random convergence
        for _i in range(5):
            fig, axs = plt.subplots(4, 4, figsize=(15, 15))  

            for i, OD in enumerate(OD_sample):
                row, col = divmod(i, 4)  

                x_values = np.arange(1, len(error_od[OD]) + 1)
                error = np.array(error_od[OD])
                error = np.abs(error)

                axs[row, col].plot(x_values, error, linestyle='-', linewidth=3)
                axs[row, col].set_xlabel('Iterations', fontsize=20)
                axs[row, col].set_ylabel('Bound gap', fontsize=20)
                axs[row, col].set_title(f"OD: {OD}", fontsize=22)
                axs[row, col].tick_params(axis='x', labelsize=20)
                axs[row, col].tick_params(axis='y', labelsize=20)
                # axs[row, col].grid(True)
            plt.tight_layout()
            plt.savefig(f'{folder}/Multiplots/Convergence_{_i}.pdf')
            plt.close()
        
        shutil.copy(f'{folder}/Multiplots/Convergence_{_i}.pdf', f'./Plots/paperPlots/{networkname}_Convergence_{_i}.pdf')

    except ValueError:
        pass

    all_OD = list(optimalSol.keys())
    all_OD = [OD for OD in all_OD if OD not in stochasticUE.od_with_no_assignments]

    convergence_rate = {}
    for i, OD in enumerate(all_OD):
        error = np.array(error_od[OD])  
        iterations = np.arange(1, len(error) + 1)

        nonzero_indices = error > 0
        iterations = iterations[nonzero_indices]
        error = error[nonzero_indices]


        if len(iterations) > 2:
            # Compute empirical convergence rate |x_{i+1}| / |x_i|
            ratios = np.abs(error[1:] / error[:-1])

            num_points = max(1, len(ratios) // 4)
            ratios_subset = ratios[-num_points:]

            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)

            mean_ratio_end = np.mean(ratios_subset)
            mean_std_ratio = np.std(ratios_subset)

            convergence_rate[OD] = (mean_ratio, std_ratio, mean_ratio_end, mean_std_ratio)

    df = pd.DataFrame(convergence_rate, index=['Mean', 'SD', 'Mean_end', 'SD_end']).T
    df.to_csv(f'{folder}/ConvergenceRate.csv')
    
    x = [rate[0] for rate in convergence_rate.values()]
    mean_rate = np.mean(x)
    std_rate = np.std(x)

    x = [rate[2] for rate in convergence_rate.values()]
    mean_rate_end = np.mean(x)
    std_rate_end = np.std(x)

    print(f"Mean rate: {mean_rate:.2f}, SD: {std_rate:.2f}")
    print(f"Mean rate end: {mean_rate_end:.2f}, SD end: {std_rate_end:.2f}")

    with open(f'{folder}/ConvergenceRate.txt', 'w') as f:
        f.write(f"Mean rate: {mean_rate:.2f}, Pooled SD: {std_rate:.2f}")
        f.write(f"\nMean rate end: {mean_rate_end:.2f}, Pooled SD end: {std_rate_end:.2f}")
        f.close()

    try:
        for _i in range(5):

            all_OD = list(optimalSol.keys())
            all_OD = [OD for OD in all_OD if OD not in stochasticUE.od_with_no_assignments]

            fig, axs = plt.subplots(4, 4, figsize=(15, 15))

            for i, OD in enumerate(OD_sample):
                row, col = divmod(i, 4)  

                error = np.array(error_od[OD])  
                iterations = np.arange(1, len(error) + 1)

                nonzero_indices = error > 0
                iterations = iterations[nonzero_indices]
                error = error[nonzero_indices]

                if len(iterations) > 2:
                    # Compute empirical convergence rate |x_{i+1}| / |x_i|
                    ratios = np.abs(error[1:] / error[:-1])

                    num_points = max(1, len(ratios) // 4)
                    ratios_subset = ratios[-num_points:]

                    mean_ratio = np.mean(ratios_subset)

                    # Plot in the corresponding subplot
                    ax = axs[row, col]
                    ax.plot(ratios, linestyle='-', label=r'$|c_{i+1}| / |c_i|$', linewidth=3)
                    ax.axhline(y=mean_ratio, color='r', linestyle='--', label=f'Mean Rate: {mean_ratio:.2f}', linewidth=2)
                    ax.set_xlabel('Iteration', fontsize=20)
                    ax.set_ylabel('Convergence rate', fontsize=20)
                    ax.set_title(f"OD: {OD}\nRate: {mean_ratio:.2f}", fontsize=22)
                    ax.tick_params(axis='x', labelsize=20)
                    ax.tick_params(axis='y', labelsize=20)
                    # ax.grid(True)
                    


            plt.tight_layout()
            plt.savefig(f'{folder}/Multiplots/ConvergenceRate_{_i}.pdf')
            plt.close()
        
        shutil.copy(f'{folder}/Multiplots/ConvergenceRate_{_i}.pdf', f'./Plots/paperPlots/{networkname}_ConvergenceRate_{_i}.pdf')

    except ValueError:
        pass

def plotAll(stochasticUE, theta, optimalSol, optimalLinkFlows, ooptimalLinkCost, optimalTSTT, neworkname, r):
    """
    Generates all the plots for the given stochasticUE object and optimal solutions.
    This function will call all the individual plotting functions and save the plots in the specified folder.
    """
    folder = f'./Plots/{neworkname.lower()}_r_{r}'

    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)  
    else:
        os.makedirs(folder)

    os.makedirs(f'{folder}/Bound')
    os.makedirs(f'{folder}/LogBound')
    os.makedirs(f'{folder}/Convergence')
    os.makedirs(f'{folder}/Multiplots')
     
    PlotLinkFlowBound(stochasticUE, optimalLinkFlows, folder, neworkname)
    PlotTTBound(stochasticUE, ooptimalLinkCost, folder, neworkname)
    PlotTSTTBound(stochasticUE, optimalTSTT, folder, neworkname)

    OD_sample = PlotBound(stochasticUE, optimalSol, folder, r, neworkname)
    plotConvergence(stochasticUE, optimalSol, folder, r, neworkname, OD_sample)

    