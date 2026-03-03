from typing import Dict, Any, Optional, List

from backend.app.visualization.schemas import VisualizationPlan, ColumnPairingPlan, PlotConfig
from backend.app.visualization.llm_utils import LLMUtils
from backend.app.visualization.profile_parser import ProfileParser
from backend.app.visualization.pairing_agent import PairingAgent
from backend.app.visualization.plot_agent import PlotAgent
from backend.app.visualization.explain_agent import ExplainAgent


class VisualizationAgent:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = LLMUtils(model=model)
        self.parser = ProfileParser()
        self.pairing = PairingAgent(self.llm)
        self.plot = PlotAgent(self.llm)
        self.explain_agent = ExplainAgent(self.llm)

    def create_plan(
        self,
        dataset_id: str,
        profile: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> VisualizationPlan:
        summary = self.parser.parse(profile)
        metrics = metrics or profile.get("_metrics") or {}

        pairing_plan = self.pairing.get_pairings(dataset_id, summary, metrics)
        plots = self.plot.get_plots(pairing_plan.pairings, summary, metrics)

        return VisualizationPlan(
            dataset_id=dataset_id,
            pairings=pairing_plan.pairings,
            plots=plots,
        )

    def get_pairings(
        self,
        dataset_id: str,
        profile: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ColumnPairingPlan:
        summary = self.parser.parse(profile)
        metrics = metrics or profile.get("_metrics") or {}
        return self.pairing.get_pairings(dataset_id, summary, metrics)

    def get_plots(
        self,
        dataset_id: str,
        profile: Dict[str, Any],
        selected_pairings: List,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> List[PlotConfig]:
        summary = self.parser.parse(profile)
        metrics = metrics or profile.get("_metrics") or {}
        return self.plot.get_plots(selected_pairings, summary, metrics)

    def explain_visualization(self, plot_title: str, axis_info: str) -> str:
        return self.explain_agent.explain(plot_title, axis_info)