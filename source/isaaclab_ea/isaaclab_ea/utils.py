from evotorch.operators import OnePointCrossOver,TwoPointCrossOver, MultiPointCrossOver, SimulatedBinaryCrossOver, GaussianMutation, PolynomialMutation

def parse_ga_operators_cfg(problem, ga_cfg, crossover_func = None, mutation_func = None):
    operators = []
    if crossover_func:
        operators.append(crossover_func)
    else:
        crossover_operator = ga_cfg['operators']['crossover']['name'].upper()
        if crossover_operator == 'ONEPOINT':
            operators.append(OnePointCrossOver(problem, **ga_cfg['operators']['crossover']['args']))
        elif crossover_operator == 'TWOPOINT':
            operators.append(TwoPointCrossOver(problem, **ga_cfg['operators']['crossover']['args']))
        elif crossover_operator == 'MULTIPOINT':
            operators.append(MultiPointCrossOver(problem, **ga_cfg['operators']['crossover']['args']))
        elif crossover_operator == 'SIMULATEDBINARY':
            operators.append(SimulatedBinaryCrossOver(problem, **ga_cfg['operators']['crossover']['args']))
        else:
            raise KeyError(f"Error: crossover operator {ga_cfg['operators']['crossover']['name']} not recognized") # TODO: change to correct one
        
    if mutation_func:
        operators.append(mutation_func)
    else:
        mutationoperator = ga_cfg['operators']['mutation']['name'].upper()
        if mutationoperator == 'GAUSSIAN':
            operators.append(GaussianMutation(problem, **ga_cfg['operators']['mutation']['args']))
        elif mutationoperator == 'POLYNOMIAL':
            operators.append(PolynomialMutation(problem, **ga_cfg['operators']['mutation']['args']))
        else:
            raise KeyError(f"Error: mutation operator {ga_cfg['operators']['mutation']['name']} not recognized") # TODO: change to correct one
    
    return operators