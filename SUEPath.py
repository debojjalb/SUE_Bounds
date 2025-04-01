import math
import time
import numpy as np
import pandas as pd
import networkx as nx
from helper import distribute_demand, dict_norm, dict_difference
from itertools import islice
from collections import deque
from numpy.linalg import norm
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        self.timers[func.__name__] += time.time() - start_time
        return result
    return wrapper

class StochasticUE:
    def __init__(self, network, theta, debug, num_paths, r, optimalSolProvided = None):
        print("Stochastic User Equilibrium Initialization...")

        self.tripSet = network.tripSet
        self.zoneSet = network.zoneSet
        self.linkSet = network.linkSet
        self.nodeSet = network.nodeSet
        self.originZones = set([k[0] for k in self.tripSet])

        # link details
        self.linkFlows = {l: 0.0 for l in self.linkSet}
        self.linkCosts = {l: 0.0 for l in self.linkSet}
        self.linkCostDerivatives = {l: 0.0 for l in self.linkSet}

        # path details
        self.theta = theta
        self.k = num_paths
        self.r = r

        self.od_with_no_assignments = []
        self.allPathsODCache = self._allPathOD()
        self.commonLinks = {OD: {str(path_i): {str(path_j): [] for path_j in self.allPathsODCache[tuple(OD)]} for path_i in self.allPathsODCache[tuple(OD)]} for OD in self.tripSet}

        self.linkLipscitz = max(self.linkSet[l].fft * self.linkSet[l].alpha * self.linkSet[l].beta / self.linkSet[l].capacity for l in self.linkSet.keys())        

        self.debug = debug
        self.tstt = None
        self.gap = 10000000

        self.currentPathCosts = {(orgn, dest): {} for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.currentPathFlows = {(orgn, dest): {} for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.targetPathFlows = {(orgn, dest): {} for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.logitProbablities = {(orgn, dest): {} for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.linkPathIncidence = {(orgn, dest): 0 for orgn in self.originZones for dest in self.zoneSet[orgn].destList}

        # For bound matrix
        
        self.boundMatrixOverIts = {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.lambdaMaxOverIts = {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        
        self.linkBoundOverIts = []
        self.ttboundsOverIts = []
        self.tsttBoundOverIts = []
        
        self.currentLmabda = {(orgn, dest): None for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.normDelta = {(orgn, dest): None for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.normDeltaTranspose = {(orgn, dest): None for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.linkCostsOverIts = []

        # For logging

        self.optimalSol = None
        self.optimalLinkFlows = None
        self.optimalLinkCost = None
        self.optimalTSTT = None

        self.currentFlowOverIts = []
        self.currentPathFlowsOverIts = {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.targetFlowOverIts = {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.pathFlowNormOverIts =  {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.pathCostOverIts =  {(orgn, dest): [] for orgn in self.originZones for dest in self.zoneSet[orgn].destList}
        self.linkFlowsOverIts = []
        self.linkCostsOverIts = []
        self.TSTTOverIts = []

        self.it = 0

        self.df = pd.DataFrame(columns=["iteration", "origin", "destination", "path", "current flow", "target flow", "cost", "logit probs"])
        
        # For timing
        self.timers = {
            "updatePathCosts": 0,
            "updateLogitProbablities": 0,
            "updateTargetPathFlows": 0,
            "writeUEresults": 0,
            "updateGap": 0,
            "updateLogs": 0,
            "boundMatrix": 0
        }

        # Run intiialisation functions
        self._createLinkPathIncidence()
        self._pathComLinksCache()

        print("Stochastic User Equilibrium Initialization Done")

        # If optimal solution is provided
        self.optimalSolProvided = optimalSolProvided
   
    """
    Implicit functions
    """
    def _createLinkPathIncidence(self):
        """
        Creates a link path incidence matrix
        """
        links = list(self.linkSet.keys())
        numLinks = len(links)
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                paths = self.allPathsODCache[orgn, dest]
                nummPaths = len(paths)
                A = np.zeros((nummPaths, numLinks))

                for i, path in enumerate(paths):
                    pathLinks = [(path[j], path[j+1]) for j in range(len(path)-1)]
                    for link in pathLinks:
                        j = links.index(link)
                        A[i][j] = 1
                self.linkPathIncidence[orgn, dest] = A
                self.normDelta[orgn, dest] = np.linalg.norm(A, 2)
                self.normDeltaTranspose[orgn, dest] = np.linalg.norm(A.T, 2)

    def _pathComLinksCache(self):
        """
        Saves the common links between paths for each OD pair in a cache
        """
        for OD in self.tripSet:
            for path_i in self.allPathsODCache[OD]:
                for path_j in self.allPathsODCache[OD]:

                    links_i = [(path_i[k], path_i[k+1]) for k in range(len(path_i)-1)]
                    links_j = [(path_j[k], path_j[k+1]) for k in range(len(path_j)-1)]

                    common_links = set(links_i).intersection(set(links_j))
                    self.commonLinks[OD][str(path_i)][str(path_j)] = common_links

    def _linkCost(self, link, flow):
        """
        Updates the link costs for a given flow
        """

        if self.debug == False:
            return self.linkSet[link].fft*(1 + self.linkSet[link].alpha*math.pow((flow*1.0/self.linkSet[link].capacity), self.linkSet[link].beta)) + self.linkSet[link].tollInTime
        else:
            if link == ('1', '2'):
                return 10 * flow
            elif link == ('1', '3'):
                return 50 + flow
            elif link == ('2', '3'):
                return 10 + flow
            elif link == ('2', '4'):
                return 50 + flow
            elif link == ('3', '4'):
                return 10 * flow
            else:
                print("Invalid link")
                quit()
    
    def _linkCostDerivative(self, link, flow):
        """
        Updates the link cost derivatives for a given flow
        """
        if self.debug == False:
            return self.linkSet[link].fft * self.linkSet[link].alpha * self.linkSet[link].beta * math.pow(flow / self.linkSet[link].capacity, self.linkSet[link].beta - 1) / self.linkSet[link].capacity
        else:
            if link == ('1', '2'):
                return 10
            elif link == ('1', '3'):
                return 1
            elif link == ('2', '3'):
                return 1
            elif link == ('2', '4'):
                return 1
            elif link == ('3', '4'):
                return 10
            else:
                print("Invalid link")
                quit()
        
    def _updateLinkFlows(self):
        """
        Updates the link flows based on the current path flows
        """
        for link in self.linkSet:
            self.linkFlows[link] = sum([self.currentPathFlows[orgn, dest][str(path)] for orgn in self.originZones for dest in self.zoneSet[orgn].destList for path in self.allPathsODCache[orgn, dest] if link in zip(path, path[1:])])
    
    def _updateLinkCosts(self):
        """
        Updates the link costs for current link flows
        """
        for link in self.linkSet:
            self.linkCosts[link] = self._linkCost(link, self.linkFlows[link])

    def _updateLinkCostDerivatives(self):
        """
        Updates the link cost derivatives for current link flows
        """
        for link in self.linkSet:
            self.linkCostDerivatives[link] = self._linkCostDerivative(link, self.linkFlows[link])

    def _allPathOD(self):
        """
        Creates a cache of all paths for each OD pair using NetworkX's shortest_simple_paths function.
        """
        G = nx.DiGraph()
        for l in self.linkSet:
            try:
                G.add_edge(self.linkSet[l].tailNode, self.linkSet[l].headNode)
            except ValueError:
                print("Graph is not created, check the node names. Node names should be integers.")
                print("Link ", l, " has tail node ", self.linkSet[l].tailNode, " and head node ", self.linkSet[l].headNode, "after conversion to int it is ", int(self.linkSet[l].tailNode), " and ", int(self.linkSet[l].headNode))
                quit()
        allPathOD = {}

        for orgn, zone in  self.zoneSet.items():
            for dest in zone.destList:
                allPathOD[orgn, dest] = list(islice(nx.shortest_simple_paths(G, orgn, dest), self.k))
                # if len(allPathOD[orgn, dest]) < self.k and orgn != dest:
                #     print(f"OD pair: {orgn}, {dest} with less than {self.k} paths ({len(allPathOD[orgn, dest])})")
                if len(allPathOD[orgn, dest]) <=1 or orgn == dest or self.tripSet[orgn, dest].demand == 0:
                    self.od_with_no_assignments.append((orgn, dest))
        print(f"OD pairs with no assignments: {len(self.od_with_no_assignments)} out of {len(self.tripSet)}")
        return allPathOD
    
    """
    For Assignment
    """
    
    def updateCurrentPathFlows(self, alpha):
        """
        Updates the current path flows using the target path flows using MSA (Method of Successive Averages)
        """
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                for path in self.allPathsODCache[orgn, dest]:
                    self.currentPathFlows[orgn, dest][str(path)] = (1-alpha)*self.currentPathFlows[orgn, dest][str(path)] + (alpha)*self.targetPathFlows[orgn, dest][str(path)]
    
    def updatePathCosts(self):
        """
        This function updates the path costs for each path in the current iteration.
        1. It first updates the link flows, costs and derivatives
        2. Then it computes the path costs for each path in the allPathsODCache
        """

        self._updateLinkFlows()
        self._updateLinkCosts()
        self._updateLinkCostDerivatives()
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                A = self.linkPathIncidence[orgn, dest]
                link_costs = np.array(list(self.linkCosts.values()))
                path_costs = A @ link_costs
                for i, path in enumerate(self.allPathsODCache[orgn, dest]):
                    self.currentPathCosts[orgn, dest][str(path)] = path_costs[i] 
        self.tstt = sum([self.linkCosts[link]*self.linkFlows[link] for link in self.linkSet])

    def updateLogitProbablities(self):
        '''
        This method updates the logit probabilities for each path
        '''
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                max_pathTT = max(-self.theta * self.currentPathCosts[orgn, dest][str(path)] for path in self.allPathsODCache[orgn, dest])
                log_sum_exp = math.log(sum(math.exp(-self.theta * self.currentPathCosts[orgn, dest][str(path)] - max_pathTT) for path in self.allPathsODCache[orgn, dest]))
                for path in self.allPathsODCache[orgn, dest]:
                    self.logitProbablities[orgn, dest][str(path)] = math.exp(-self.theta * self.currentPathCosts[orgn, dest][str(path)] - max_pathTT - log_sum_exp)

    def updateTargetPathFlows(self):
        '''
        This method updates the target path flows
        '''
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                for path in self.allPathsODCache[orgn, dest]:
                    self.targetPathFlows[orgn, dest][str(path)] = self.logitProbablities[orgn, dest][str(path)] * self.tripSet[orgn, dest].demand

    def iteration0(self, initType, networkName):
        '''
        This method initializes the path flows and path costs for the first iteration
        '''
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                numPaths = len(self.allPathsODCache[orgn, dest])
                # distribute demand ranmly into n parts so that the sum of the parts is equal to the demand
                if initType == 'equal':
                    distribution = [self.tripSet[orgn, dest].demand/numPaths for _ in range(numPaths)]
                elif initType == 'random':
                    distribution = distribute_demand(self.tripSet[orgn, dest].demand, numPaths)         
                elif initType == 'logitFreeFlow':
                    distribution = [0 for _ in range(numPaths)]
                    distribution[0] = self.tripSet[orgn, dest].demand           
                else:
                    print("Invalid initialization method")
                    quit()
                for i, path in enumerate(self.allPathsODCache[orgn, dest]):
                    self.currentPathFlows[orgn, dest][str(path)] = distribution[i]
                    self.targetPathFlows[orgn, dest][str(path)] =  distribution[i]
        print("Updating path costs")
        self.updatePathCosts() 

    def updateGap(self):
        '''
        This method updates the gap between the current and target path flows
        '''
        gap = 0
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                for path in self.allPathsODCache[orgn, dest]:
                    gap += abs(self.currentPathFlows[orgn, dest][str(path)] - self.targetPathFlows[orgn, dest][str(path)])
        self.gap = gap

    def assignment(self, accuracy, maxIter, initType, calBounds, networkName):
        """
        Performs the Stochastic User Equilibrium assignment algorithm.
        """

        self.iteration0(initType, networkName)
        it = 1
        converged = False
        startP = time.time()

        prev_gap = deque(maxlen=3)
        epsilon = 0.01

        start_time = time.time()

        for it in range(1, maxIter):
            
            self.it = it
            if it <= 10:
                alpha = 1 / it
            else:
                if abs(prev_gap[-1] - prev_gap[0]) / prev_gap[0] < epsilon:
                    alpha = 1 / it
                    print(f"Reducing alpha")

            # alpha = 1 / it
                
            self.updateCurrentPathFlows(alpha)
            self.updatePathCosts()
            self.updateLogitProbablities()
            self.updateTargetPathFlows()
            self.updateGap()

            # If testing bound accuracy
            self.testBounds()

            prev_gap.append(self.gap)

            print(f"Current gap: {self.gap} and time elapsed: {time.time() - start_time}")
            self.optimalSol = self.currentPathFlows
            self.optimalLinkFlows = self.linkFlows
            self.optimalLinkCost = self.linkCosts
            self.optimalTSTT = self.tstt

            if calBounds and it >= 1:
                self.updateLogs()
                self.boundMatrix()
                self.linkBound()
                self.ttbound()
                self.tsttBound()

            if self.gap < accuracy:
                print("Assignment took", time.time() - startP, " seconds")
                print("Assignment converged in ", it, " iterations")
                converged = True
                break

            # if time.time() - startP > 60*10:
            #     return
            
            # avg_time_first_five_iterations.append(time.time() - startP)
            # startP = time.time()
            # if it >= 5:
            #     print("\n\nAverage time for first five iterations: ", sum(avg_time_first_five_iterations) / len(avg_time_first_five_iterations))
            #     return

        if not converged:
            print("The assignment did not converge with the desired gap and max iterations are reached")
            print("Current gap: ", self.gap)


    """
    For logging
    """
    
    def updateLogs(self):
        for orgn in self.originZones:
            for dest in self.zoneSet[orgn].destList:
                self.pathCostOverIts[orgn, dest].append(self.currentPathCosts[orgn, dest].copy())
                self.currentPathFlowsOverIts[orgn, dest].append(self.currentPathFlows[orgn, dest].copy())
                self.targetFlowOverIts[orgn, dest].append(self.targetPathFlows[orgn, dest].copy())
                self.pathFlowNormOverIts[orgn, dest].append(dict_norm(self.currentPathFlows[orgn, dest]))
        self.linkCostsOverIts.append(self.linkCosts.copy())
        self.linkFlowsOverIts.append(self.linkFlows.copy())
        self.TSTTOverIts.append(self.tstt)
        self.currentFlowOverIts.append(self.linkFlows.copy())

    """
    For Bounds
    """

    def logitJacobian(self, p):
        n = len(p) 
        J = np.zeros((n, n))
        for j in range(n):
            for i in range(j):
                J[i, j] = self.theta * p[i] * p[j]
            J[j, j] = - self.theta * p[j] * (1 - p[j])
        return J
       
    def netJacobian(self, OD):
        jacobian_matrix = np.zeros((len(self.allPathsODCache[OD]), len(self.allPathsODCache[OD])))
        for path_i in self.allPathsODCache[OD]:
            for path_j in self.allPathsODCache[OD]:
                common_links = self.commonLinks[OD][str(path_i)][str(path_j)]
                sum_ = 0
                for link in common_links:
                    sum_ += self.linkCostDerivatives[link]
                jacobian_matrix[self.allPathsODCache[OD].index(path_i)][self.allPathsODCache[OD].index(path_j)] = sum_
        return jacobian_matrix
        
    def targetJacobian(self, OD):
        d = self.tripSet[OD].demand
        costJacobian = self.netJacobian(OD)
        probJacobian = self.logitJacobian(list(self.logitProbablities[OD].values()))
        hStarJacobian = d * probJacobian 
        targetJacobian = hStarJacobian @ costJacobian
        return targetJacobian
    
    def boundMatrix(self):
        """
        calculates the path flow bounds for each iteration
        """
        optimalSol = self.optimalSolProvided

        if optimalSol is not None:
            looseness = 0

        for OD in self.tripSet:
            if self.tripSet[OD].demand == 0:
                self.boundMatrixOverIts[OD].append(np.eye(len(self.allPathsODCache[OD])))
                self.lambdaMaxOverIts[OD].append(1)
                self.currentLmabda[OD] = 1
                continue
            else:
                A = self.targetJacobian(OD)
            I = np.eye(A.shape[0])

            bound = np.linalg.inv(I - A)

            self.boundMatrixOverIts[OD].append(bound)
            # print(bound)
            bound_lambda = np.linalg.norm(bound, 2) / (1-self.r)
            self.currentLmabda[OD] = bound_lambda
            self.lambdaMaxOverIts[OD].append(bound_lambda)

            # To check the tightness of the bound
            h_i = self.currentPathFlowsOverIts[OD][-1]
            h_next = self.targetFlowOverIts[OD][-1]
            h_star = optimalSol[OD]

            if optimalSol is not None:
                bound_val = (abs(bound_lambda) * norm(np.array(list(dict_difference(h_next, h_i).values()))))
                actual_val = (norm(np.array(list(dict_difference(h_star, h_i).values()))))
                looseness += abs(bound_val - actual_val)
            
        if optimalSol is not None:
            print("Bound gap is", looseness)

            
    def testBounds(self):
        """
        Checks if the bounds are satisfied
        """
        bound_satisfied = True

        Bound_not_satisfied_OD ={}

        if self.it < 2:
            return

        for OD in self.tripSet:
            if self.tripSet[OD].demand == 0:
                continue

            h_star = self.optimalSolProvided[OD]
            h_i = self.currentPathFlows[OD]
            h_next = self.targetPathFlows[OD]
            bound_lambda = self.currentLmabda[OD]

            actual_val = norm(np.array(list(dict_difference(h_star, h_i).values())))
            bound_val = abs(bound_lambda) * norm(np.array(list(dict_difference(h_next, h_i).values())))
            # print(f"OD: {OD}, Actual: {actual_val}, Bound: {bound_val}")
            if actual_val > bound_val:
                bound_satisfied = False
                Bound_not_satisfied_OD[OD] = (bound_val - actual_val)
                # print("Bound not satisfied", actual_val, bound_val)

        if bound_satisfied:
            print(f"{self.it}: Bound satisfied")
            # print("Iteration: ", self.it)
            # print("Gap: ", self.gap)
            # quit()
        else:
            print(f"{self.it}: Bound not satisfied in {len(Bound_not_satisfied_OD)} OD out of {len(self.tripSet)}")
            # print(Bound_not_satisfied_OD)
            

    def linkBound(self):
        """
        calculates the total travel time bound
        """
        linkbound = 0
        for OD in self.tripSet:
            lambdaMax = self.currentLmabda[OD]
            deltaNorm = self.normDelta[OD]
            linkbound += lambdaMax * deltaNorm * dict_norm(dict_difference(self.currentPathFlows[OD], self.targetPathFlows[OD]))

        self.linkBoundOverIts.append(linkbound)

    def ttbound(self):
        """
        calculates the total travel time bound
        """
        ttbound = self.linkLipscitz * self.linkBoundOverIts[-1]
        self.ttboundsOverIts.append(ttbound)
            
    def tsttBound(self):
        """
        calculates the total system travel time bounds
        """
        m_1 = self.linkBoundOverIts[-1]
        m_2 = self.ttboundsOverIts[-1]

        mod_x = dict_norm(self.linkFlows)
        mod_tx = dict_norm(self.linkCosts)

        tsttbound = (m_1 + mod_x) * (m_2 + mod_tx) - self.tstt

        self.tsttBoundOverIts.append(tsttbound)



    
    