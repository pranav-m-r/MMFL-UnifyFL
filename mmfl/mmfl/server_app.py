"""mmfl: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from mmfl.task import get_weights
from mmfl.models import CombinedModel
from typing import List, Tuple, Optional, Dict
import numpy as np

class CustomFedAvg(FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[List[np.ndarray], int]],
        failures: List[BaseException],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, float]]:
        # Aggregate feature extractors and classifiers separately
        feature_extractor1_weights = []
        feature_extractor2_weights = []
        classifier_weights = []
        num_examples_total = sum(num_examples for _, num_examples in results)

        for weights, num_examples in results:
            feature_extractor1_weights.append([layer * num_examples for layer in weights[:5]])
            feature_extractor2_weights.append([layer * num_examples for layer in weights[5:10]])
            classifier_weights.append([layer * num_examples for layer in weights[10:]])

        aggregated_feature_extractor1 = [
            sum(layer_updates) / num_examples_total for layer_updates in zip(*feature_extractor1_weights)
        ]
        aggregated_feature_extractor2 = [
            sum(layer_updates) / num_examples_total for layer_updates in zip(*feature_extractor2_weights)
        ]
        aggregated_classifier = [
            sum(layer_updates) / num_examples_total for layer_updates in zip(*classifier_weights)
        ]

        aggregated_weights = aggregated_feature_extractor1 + aggregated_feature_extractor2 + aggregated_classifier
        return aggregated_weights, {}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(CombinedModel())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)