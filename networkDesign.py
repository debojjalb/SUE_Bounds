from readNetwork import Network
from SUEPath import StochasticUE
from genPlots import *
from numpy.linalg import norm
from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat
import time
import pandas as pd

USE_BOUNDS = False # Set to True to use bounds on the optimal link costs, otherwise False to use the optimal solution directly
OPTIMAL_SOL_ACCURACY = 0.001
BOUND_ACCURACY = 0.001
R = 0.3

input_location_list = ["testNetworks/Sioux Falls Network/", "testNetworks/EMA Network/", "testNetworks/Berlin MC Network/", "testNetworks/Anaheim Network/"]
inputLocation = input_location_list[0]
networkName = str(inputLocation.split("/")[1]).split(" ")[0]
print("Network:", networkName)
convert_network_tntp_to_dat(f'{inputLocation}/net.tntp', f'{inputLocation}/network.dat')
convert_demand_tntp_to_dat(f'{inputLocation}/trips.tntp', f'{inputLocation}/demand.dat')

network = Network()
network.readNetwork(inputLocation)
network.readDemand(inputLocation)

def runSUE(accuracy, calBounds):
    debug = False
    theta = 0.5
    k = 10 
    initType = 'logitFreeFlow'
    r = R

    optSolAccuracy = accuracy
    optSolMaxIter = 100000

    stochasticUE = StochasticUE(network, theta, debug, k, r)
    stochasticUE.assignment(accuracy=optSolAccuracy, maxIter=optSolMaxIter, initType=initType, calBounds = calBounds, networkName=networkName)

    optimalLinkCosts = stochasticUE.optimalLinkCost
    norm_optimalLinkCosts = norm(np.array(list(optimalLinkCosts.values())))

    optimalTSTT = stochasticUE.optimalTSTT

    # bound on the optimal link costs
    if calBounds:
        bounds_on_link_cost = stochasticUE.ttboundsOverIts[-1]
        bounds_on_tstt = stochasticUE.tsttBoundOverIts[-1]
    else:
        bounds_on_link_cost = 0 # No upper bound if not calculating bounds
        bounds_on_tstt = 0 # No upper bound if not calculating bounds
    
    return norm_optimalLinkCosts, bounds_on_link_cost, optimalTSTT, bounds_on_tstt


print("Network:", networkName)

best_link_to_toll = None
best_TSTT = 10000000000000
start_time = time.time()

df = pd.DataFrame(columns = ['Link to toll', 'TSTT', 'Bound on TSTT', 'Estimaed eqm. TSTT using bounds', 'Time taken so far'])
for link_to_toll in network.linkSet:
    print("Link to toll:", link_to_toll)
    # impose toll
    network.linkSet[link_to_toll].tollInTime = 100 * 60 / 240 # 240 v.o.t and a 10$ toll

    # run SUE
    if USE_BOUNDS == False:
        _, _, optimal_TSTT, bounds_on_tstt = runSUE(accuracy = OPTIMAL_SOL_ACCURACY, calBounds = False)
        print("TSTT:", optimal_TSTT)
        print("Bounds on TSTT:", bounds_on_tstt)
        if optimal_TSTT < best_TSTT:
            best_TSTT = optimal_TSTT
            best_link_to_toll = link_to_toll
    else:
        _, _,  optimal_TSTT, bounds_on_tstt = runSUE(accuracy = BOUND_ACCURACY, calBounds = True)
        print("TSTT:", optimal_TSTT)
        print("Bounds on TSTT:", bounds_on_tstt)
        if optimal_TSTT + bounds_on_tstt < best_TSTT:
            best_TSTT = optimal_TSTT + bounds_on_tstt 
            best_link_to_toll = link_to_toll

    # release toll
    network.linkSet[link_to_toll].tollInTime = 0.0
    new_row = pd.DataFrame([{
        'Link to toll': link_to_toll,
        'TSTT': optimal_TSTT,
        'Bound on TSTT': bounds_on_tstt,
        'Estimaed eqm. TSTT using bounds': optimal_TSTT + bounds_on_tstt,
        'Time taken so far': round(time.time() - start_time, 2)
    }])

    # Append using concat
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(f'netDesRes/TSTT_results_{USE_BOUNDS}_{networkName}_{BOUND_ACCURACY}.csv')

    print("\n\n")
    print("Results so far:")
    print("Best link to toll:", best_link_to_toll)
    print("Best TSTT:", best_TSTT)
    print("Time taken so far:", round(time.time() - start_time, 2), " seconds")
    print("\n\n")

# Print the least 5 link to tolls
print("Least 5 link to tolls:")
print(df.sort_values(by=['Estimaed eqm. TSTT using bounds']).head(5))
print("\n\n")


