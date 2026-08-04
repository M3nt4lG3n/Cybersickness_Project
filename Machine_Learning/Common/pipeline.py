"""
pipeline.py

Execution engine for the Cybersickness ML framework.

A Pipeline consists of an ordered list of PipelineStages.
Each stage receives an Experiment object, modifies it,
and returns it for the next stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import time

from .experiment import Experiment

class PipelineStage(ABC):
    """
    Base class for every stage in the ML pipeline.

    Stages should be stateless whenever possible.
    """

    stage_name = "Unnamed"

    def validate(self, experiment: Experiment):

        pass

    @abstractmethod
    def run(self, experiment: Experiment) -> Experiment:
        """
        Execute the stage.

        Parameters
        ----------
        experiment
            Current experiment state.

        Returns
        -------
        Experiment
            Updated experiment.
        """
        raise NotImplementedError

class Pipeline:
    """
    Executes an ordered sequence of pipeline stages.
    """

    def __init__(self):

        self.stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage):

        self.stages.append(stage)

        return self

    def extend(self, stages: Iterable[PipelineStage]):

        self.stages.extend(stages)

        return self
    
    def run(self, experiment):

        pipeline_start = time.perf_counter()

        for stage in self.stages:

            stage_start = time.perf_counter()

            experiment = stage.run(experiment)

            experiment.runtime.stage_times[stage.stage_name] = (
                time.perf_counter() - stage_start
            )

        experiment.runtime.total_runtime = (
            time.perf_counter() - pipeline_start
        )

        return experiment

class PipelineFactory:

    @staticmethod
    def create_standard():

        return (
            Pipeline()

            .add_stage(DataLoaderStage())

            .add_stage(PreprocessingStage())

            .add_stage(FeatureSelectionStage())

            .add_stage(ModelSelectionStage())

            .add_stage(EvaluationStage())

            .add_stage(PlottingStage())

            .add_stage(SaveResultsStage())
        )