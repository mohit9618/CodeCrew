from backend.app.agents.developer import DeveloperAgent


class FrontendDeveloper(DeveloperAgent):

    def __init__(self):
        super().__init__(specialization="frontend")