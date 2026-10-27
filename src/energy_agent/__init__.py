from energy_agent.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"energy-agent: provider={settings.llm_provider}, model={settings.active_model}")
