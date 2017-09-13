import itertools
import math
import random

try:
    import draw
except ImportError:
    draw = None
from dmprsim import MobilityArea, MobilityModel
from topologies.utils import GenericTopology


class SquareTopology(GenericTopology):
    NAME = 'square'
    SIMULATION_TIME = 300
    SIZE = 3
    DEFAULT_RAND_SEED = 1

    def __init__(self,
                 simulation_time=SIMULATION_TIME,
                 size=SIZE,
                 random_seed_prep=DEFAULT_RAND_SEED,
                 random_seed_runtime=DEFAULT_RAND_SEED,
                 visualize=True,
                 simulate_forwarding=True,
                 tracepoints=(),
                 log_directory=None,
                 diagonal=False,
                 range_factor=1,
                 name=NAME,
                 config={},
                 ):
        super(SquareTopology, self).__init__(
            simulation_time,
            random_seed_runtime,
            simulate_forwarding,
            visualize,
            log_directory,
            tracepoints,
            name,
            config,
        )
        self.size = size
        self.interfaces = [
            {"name": "wifi0", "range": 0, "bandwidth": 8000, "loss": 10},
        ]
        self.random_seed_prep = random_seed_prep
        self.diagonal = diagonal
        self.range_factor = range_factor

        self.routers = []
        self.tx_router = None
        self.rx_ip = None
        self.area = None

    def prepare(self):
        random.seed(self.random_seed_prep)
        if self.visualize and draw:
            draw.setup_img_folder(self.log_directory)

        # Set all models on a circle
        padding = 25
        distance = 5

        canvas_size = self.size * distance
        if canvas_size < 400:
            canvas_size = 400
            distance = 400 // self.size
        canvas_size += padding * 2
        self.area = MobilityArea(canvas_size, canvas_size)

        range_ = distance
        if self.diagonal:
            range_ *= math.sqrt(2)
        self.interfaces[0]['range'] = range_ * self.range_factor + 1

        models = []
        if self.size == 1:
            size_x, size_y = 2, 1
        else:
            size_x, size_y = self.size, self.size

        for x, y in itertools.product(range(size_x), range(size_y)):
            models.append(MobilityModel(self.area,
                                        x=x * distance + padding,
                                        y=y * distance + padding))

        self.routers = self._generate_routers(models)

        if self.simulate_forwarding:
            self.tx_router = random.choice(self.routers)
            while True:
                rx_router = random.choice(self.routers)
                if rx_router != self.tx_router:
                    break

            self.tx_router.is_transmitter = True
            rx_router.is_receiver = True
            self.rx_ip = rx_router.pick_random_configured_network()

        return self.routers


if __name__ == '__main__':
    simulation = SquareTopology()
    simulation.prepare()
    for _ in simulation.start():
        pass