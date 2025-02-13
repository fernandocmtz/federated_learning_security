import flwr as fl
import numpy as np
from statistics import median

def median_aggregation(results):
    aggregated_params = []
    for i in range(len(results[0])):
        param_set = [client[i] for client in results]
        aggregated_params.append(np.median(param_set, axis=0))
    return aggregated_params

class FLServer(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = median_aggregation([res[1] for res in results])
        return aggregated_parameters, {}

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5), strategy=FLServer())
