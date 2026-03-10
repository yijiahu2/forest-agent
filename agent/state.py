from typing import TypedDict, Dict, Any, List


class AgentState(TypedDict, total=False):
    base_config_path: str
    generated_config_path: str
    run_name: str
    iteration: int
    max_iterations: int

    prior_summary: Dict[str, Any]
    best_history: List[Dict[str, Any]]
    latest_metrics: Dict[str, Any]
    details_summary: Dict[str, Any]

    proposal_raw: str
    proposal: Dict[str, Any]

    run_success: bool
    continue_search: bool
    final_summary: Dict[str, Any]