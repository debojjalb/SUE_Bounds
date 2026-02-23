from readNetwork import Network
from SUEPath import StochasticUE
from genPlots import *
import pickle
from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat

def run(inputLocation, plot, calBounds, k=10, theta=0.5, r=0.5, debug = False):
    """
    Runs the traffic assignment for a given network location and checks the bounds on each feasible solution.
    :param inputLocation: str, path to the network files
    :param plot: bool, whether to generate plots or not
    :param calBounds: bool, whether to calculate bounds on the optimal solution
    :param k: int, number of iterations for the stochastic assignment (default=10)
    :param theta: float, parameter for the stochastic assignment (default=0.5)
    :param r: float, parameter for the stochastic assignment (default=0.4)
    :return: None, but will save the optimal solutions and generate plots if specified
    """

    networkName = str(inputLocation.split("/")[1]).split(" ")[0]
    print("Network:", networkName)

    # Convert files
    try:
        convert_network_tntp_to_dat(f'{inputLocation}/net.tntp', f'{inputLocation}/network.dat')
        convert_demand_tntp_to_dat(f'{inputLocation}/trips.tntp', f'{inputLocation}/demand.dat')
    except FileNotFoundError:
        print(f"Conversion of TNTP files to DAT files failed for {inputLocation}. Hoping TNTP files exist and using them directly.")
        pass

    network = Network()
    network.readNetwork(inputLocation)
    network.readDemand(inputLocation)
    initType = 'logitFreeFlow'

    optSolAccuracy = 0.01
    optSolMaxIter = 100000

    boundAccuracy = 1
    boundMaxIter = 100000

    if __name__ == "__main__":
        try:
            with open(f'./optimalSolutions/{networkName}_optimalSol_{theta}_{k}_{optSolAccuracy}.pkl', 'rb') as f:
                print('Loading optimal solution from pickle')
                optimalSol = pickle.load(f)
            with open(f'./optimalSolutions/{networkName}_optimalLinkFlows_{theta}_{k}_{optSolAccuracy}.pkl', 'rb') as f:
                optimalLinkFlows = pickle.load(f)
            with open(f'./optimalSolutions/{networkName}_ooptimalLinkCost_{theta}_{k}_{optSolAccuracy}.pkl', 'rb') as f:
                ooptimalLinkCost = pickle.load(f)
            with open(f'./optimalSolutions/{networkName}_optimalTSTT_{theta}_{k}_{optSolAccuracy}.pkl', 'rb') as f:
                optimalTSTT = pickle.load(f)
            print('Optimal solution loaded successfully from pickle\n\n')
            
        except Exception as e:
            print(f'\n\nOptimal solution pickle not found. Running the assignment\n\n')
            stochasticUE = StochasticUE(network, theta, debug, k, r)
            stochasticUE.assignment(accuracy=optSolAccuracy, maxIter=optSolMaxIter, initType=initType, calBounds = False, networkName=networkName)
            optimalSol = stochasticUE.optimalSol 
            optimalLinkFlows = stochasticUE.optimalLinkFlows
            ooptimalLinkCost = stochasticUE.optimalLinkCost
            optimalTSTT = stochasticUE.optimalTSTT
            with open(f'optimalSolutions/{networkName}_optimalSol_{theta}_{k}_{optSolAccuracy}.pkl', 'wb') as f:
                pickle.dump(optimalSol, f)
            with open(f'optimalSolutions/{networkName}_optimalLinkFlows_{theta}_{k}_{optSolAccuracy}.pkl', 'wb') as f:
                pickle.dump(optimalLinkFlows, f)
            with open(f'optimalSolutions/{networkName}_ooptimalLinkCost_{theta}_{k}_{optSolAccuracy}.pkl', 'wb') as f:
                pickle.dump(ooptimalLinkCost, f)
            with open(f'optimalSolutions/{networkName}_optimalTSTT_{theta}_{k}_{optSolAccuracy}.pkl', 'wb') as f:
                pickle.dump(optimalTSTT, f)

        print("Generating Bound")
        stochasticUE = StochasticUE(network, theta, debug, k, r, optimalSol)
        stochasticUE.assignment(accuracy=boundAccuracy, maxIter=boundMaxIter, initType=initType, calBounds = calBounds, networkName=networkName)

        if plot:
            print("Generating Plots")
            plotAll(stochasticUE, theta, optimalSol, optimalLinkFlows, ooptimalLinkCost, optimalTSTT, networkName, r)


if __name__ == "__main__":
    input_location_list = ["testNetworks/Sioux Falls Network/", "testNetworks/Berlin MC Network/", "testNetworks/EMA Network/", "testNetworks/Anaheim Network/"]
    # input_location_list = ["testNetworks/Sioux Falls Network/"]
    for inputLocation in input_location_list:
        run(inputLocation, plot = False, calBounds = True)