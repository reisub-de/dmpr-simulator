"""
Generate one plot per `chartgroup` with `xaxis` as the x-axis and message size
as the y-axis
"""

import logging
import multiprocessing
from pathlib import Path

import functools
import matplotlib
import pandas as pd

#matplotlib.use('AGG')

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

import numpy as np

from dmprsim.analyze._utils.process_messages import process_files
from dmprsim.scenarios.message_size import MessageSizeScenario

configs = {
    'density': {
        'label': "Interface Range",
        'datapoints': {
            1: "2 - 4 Neighbors",
            2: "3 - 8 Neighbors",
            4: "5 - 12 Neighbors",
            5: "7 - 20 Neighbors",
            8: "8 - 24 Neighbors",
        },
    },
    'loss': {
        'label': "Loss",
        'datapoints': {
            0: "0%",
            1: "1% loss",
            2: "2% loss",
            5: "5% loss",
            10: "10% loss",
            20: "20% loss",
        },
    },
    'size': {
        'label': "Network size",
        'datapoints': {
            i: "{} Nodes".format(i ** 2) for i in (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            13,
            15,
        )},
    },
    'interval': {
        'label': "Full update interval",
        'datapoints': {
            i: "Update interval {}".format(i) for i in (
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            9,
            11,
            13,
            15,
        )},
    },
}

MAX = ('Maximum', np.max)
P90 = ('90% Percentile', functools.partial(np.percentile, q=90))
P75 = ('75% Percentile', functools.partial(np.percentile, q=75))
P25 = ('75% Percentile', functools.partial(np.percentile, q=25))
AVG = ('Average', np.average)
MEDIAN = ('Median', np.median)
MIN = ('Minimum', np.min)

# compression options: 'none', 'zlib', 'lzma
PLOTS = {
    'density': (
        ('size', {
            'interval': '*',
            'loss': '*',
            'actions': ['zlib', 'len'],
            'options': [MAX, P75, AVG, MEDIAN, P25, MIN],
        }),
        ('interval', {
            'loss': '*',
            'size': '*',
            'actions': ['zlib', 'len'],
            'options': [P75, AVG, MEDIAN, P25],
        }),
    ),
    'loss': (
        ('interval', {
            'density': '*',
            'size': '*',
            'actions': ['zlib', 'len'],
            'options': [P75, AVG, MEDIAN, P25]
        }),
    ),
    'interval': (
        ('size', {
            'loss': '*',
            'density': '*',
            'actions': ['zlib', 'len'],
            'options': [MAX, P90, P75, AVG, MEDIAN]
        }),
        # ('size', {
        #     'loss': '*',
        #     'density': '*',
        #     'actions': ['zlib', 'len'],
        #     'type': 'histogram',
        #     'options': [],
        # }),
    )
}

logger = logging.getLogger(__name__)


def accumulate(input: Path, globs: dict, filename: str) -> np.array:
    """
    for `glob`/filename in input directory, accumulate the message lengths
    and return a numpy array
    """
    files = '{size}-{density}-{loss}-{interval}/{filename}'.format(
        filename=filename, **globs)
    input_paths = input.glob(files)

    message_lengths = ''
    for file in input_paths:
        try:
            with file.open() as fin:
                message_lengths += fin.read()
        except FileNotFoundError as e:
            logger.debug("Could not find file, skipping {}".format(e))
            continue

    sizes = np.asarray([int(i) for i in message_lengths.splitlines() if i])
    if len(sizes) == 0:
        logger.debug("File lengths of {} is zero, skipping".format(files))
        return

    return pd.Series(sizes)


def plot(chartgroup: str, chartgroup_datapoint: int, xaxis: str,
         data: list, output: str, options: list):
    """
    plot the data with a defined chartgroup and xaxis with title, labels and
    a legend into the output directory with the filename
    chartgroup-chartgroup_datapoint-xaxis
    """
    x, data = zip(*data)
    if xaxis == 'size':
        x = [i ** 2 for i in x]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylabel('Message Size / bytes')
    ax.set_xlabel(configs[xaxis]['label'])
    ax.set_title(configs[chartgroup]['datapoints'][chartgroup_datapoint])

    # Fill the space between minimum and maximum
    # ax.fill_between(x, mins, maxs, color=((0.16, 0.5, 0.725, 0.31),))

    for label, func in options:
        ax.plot(x, [func(d) for d in data], label=label)

    ax.legend()

    fig.savefig(output, dpi=300)
    plt.close(fig)


def histogram(chartgroup: str, chartgroup_datapoint: int, xaxis: str,
              data: list, output: Path, options: list, filename: str):
    for x, d in data:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(d, bins=50)

        plot_filename = "histogram-{}-{}-{}-{}-{}.png".format(chartgroup,
                                                    chartgroup_datapoint, xaxis,
                                                    x, filename)
        fig.savefig(str(output / plot_filename), dpi=300)

        plt.close(fig)

def testplot(dataframe):
    dataframe.agg(['median', 'max']).plot.line()
    plt.show()

def generate_plots(input: Path, output: Path, filename: str, chartgroup: str,
                   xaxis: str, globs: dict):
    # Generate a separate chart for each datapoint in chartgroup
    try:
        output.mkdir(parents=True)
    except FileExistsError:
        pass
    for chartgroup_datapoint in configs[chartgroup]['datapoints']:
        globs[chartgroup] = chartgroup_datapoint
        cumulated_data = []
        test = {}

        # Parse all datafiles for the specified x-axis
        for xaxis_datapoint in configs[xaxis]['datapoints']:
            globs[xaxis] = xaxis_datapoint
            data = accumulate(input, globs, filename)
            if data is None:
                continue
            cumulated_data.append((xaxis_datapoint, data))
            test[xaxis_datapoint] = data

        if not cumulated_data:
            logger.debug(
                "no data for {}-{}, skippint".format(chartgroup_datapoint,
                                                     xaxis))
            continue

        type_ = globs.get('type', 'linechart')
        if type_ == 'linechart':
            plot_filename = "linechart-{}-{}-{}-{}.png".format(chartgroup,
                                                     chartgroup_datapoint,
                                                     xaxis, filename)
            plot(chartgroup, chartgroup_datapoint, xaxis, cumulated_data,
                 str(output / plot_filename), globs['options'])
        elif type_ == 'histogram':
            histogram(chartgroup, chartgroup_datapoint, xaxis, cumulated_data,
                      output, globs['options'], filename)

        testplot(pd.DataFrame(test))


def run_scenario(args: object, results_dir: Path, scenario_dir: Path):
    scenario = MessageSizeScenario(
        args,
        results_dir,
        scenario_dir,
        sizes=configs['size']['datapoints'].keys(),
        meshes=configs['density']['datapoints'].keys(),
        losses=configs['loss']['datapoints'].keys(),
        intervals=configs['interval']['datapoints'].keys(),
    )
    scenario.start()


def _process_message_worker(args):
    dir, result_file, actions = args
    process_files(dir.glob('routers/*/trace/tx.msg'), dir / result_file,
                  actions)


def process_messages(path: Path, result_file: str, actions: list):
    dirs = tuple(path.glob('*-*-*-*'))
    pool = multiprocessing.Pool()
    all_ = len(dirs)
    cur = 0
    for _ in pool.imap_unordered(_process_message_worker,
                                 ((dir, result_file, actions) for dir in dirs),
                                 chunksize=20):
        cur += 1
        logger.info('{:.2%} done'.format(cur / all_))
    pool.close()
    pool.join()


def main(args: object, results_dir: Path, scenario_dir: Path):
    if not (scenario_dir / '.done').exists():
        run_scenario(args, results_dir, scenario_dir)

    logger.info("Start plotting")
    for chartgroup in PLOTS:
        for xaxis, conf in PLOTS[chartgroup]:
            actions = conf['actions']
            result_file = '-'.join(actions)
            logger.info(
                'Processing {}-{}-{}'.format(chartgroup, xaxis, result_file))
            done_file = scenario_dir / ('.done-' + result_file)
            if not done_file.exists():
                process_messages(scenario_dir, result_file, actions)
                done_file.touch()
            generate_plots(input=scenario_dir, output=results_dir,
                           filename=result_file, chartgroup=chartgroup,
                           xaxis=xaxis, globs=conf.copy())
