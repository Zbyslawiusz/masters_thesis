import gzip
import pickle
from main_neat import Simulation


def restore_checkpoint(filename):
    """Resumes the simulation from a previous saved point."""
    with gzip.open(filename) as f:
        net = pickle.load(f)
        return net
        # return Population(config, (population, species_set, generation))


# winner_net = restore_checkpoint("./neat-models/304caba1d11140d8bd92c1f865860fe3")
winner_net = restore_checkpoint("./neat-models/4c815457d5144161ade5092c0460cc8f-2024-04-20_10-44-59")

simulation = Simulation(
        net=winner_net,
        ui_flag=True,
        number_of_links=3,
        target_xcor=2600,
        interpolation=4,
        gripper="robotic"
    )
